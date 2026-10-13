"""SmartShield AI: Streamlit UI (Home, four detectors, About).

This file only builds the page. All prediction logic lives in src/detection/.
"""

import streamlit as st

from config import settings
from src.detection.email_detector import (
    MAX_BODY_INPUT_CHARS,
    MAX_SUBJECT_CHARS,
    EmailDetector,
    load_email_detector,
)
from src.detection.model_info import summarize_models
from src.detection.phishing_detector import (
    MAX_URL_CHARS,
    PhishingDetector,
    load_phishing_detector,
)
from src.detection.rate_limit import RateLimiter
from src.detection.spam_detector import (
    LABEL_OK,
    LABEL_PROMO,
    LABEL_SCAM,
    MAX_SMS_CHARS,
    DetectorError,
    SpamDetector,
    load_sms_detector,
)
from src.detection.toxicity_detector import (
    MAX_COMMENT_CHARS,
    ToxicityDetector,
    load_toxicity_detector,
)
from src.models.toxicity_model import LABELS as TOXICITY_LABELS
from src.ui import components, dashboard, theme

st.set_page_config(page_title="SmartShield AI", page_icon="🛡️", layout="wide")
theme.inject_theme()


def too_many_requests() -> bool:
    """Per-session throttle shared by the four Analyze buttons; shows a message when it trips."""
    if "limiter" not in st.session_state:
        st.session_state["limiter"] = RateLimiter(
            settings.RATE_LIMIT_CALLS, settings.RATE_LIMIT_WINDOW_S
        )
    limiter = st.session_state["limiter"]
    if limiter.allow():
        return False
    st.warning(f"Too many requests. Please wait about {limiter.retry_after():.0f} seconds.")
    return True


def example_row(prefix: str, examples: list[tuple[str, dict]]) -> None:
    """Buttons that fill the input boxes with a harmless sample (set before the next rerun)."""
    st.caption("Try an example:")
    columns = st.columns(len(examples))
    for i, (label, values) in enumerate(examples):
        columns[i].button(
            label,
            key=f"{prefix}_example_{i}",
            on_click=st.session_state.update,
            args=(values,),
            width="stretch",
        )


@st.cache_resource(show_spinner=False)
def get_sms_detector() -> SpamDetector:
    """Load the SMS models once per server process (not on every click)."""
    return load_sms_detector()


@st.cache_resource(show_spinner=False)
def get_email_detector() -> EmailDetector:
    """Load the email model once per server process."""
    return load_email_detector()


SMS_EXPLANATIONS = {
    LABEL_OK: "This looks like an ordinary message (for example a bank alert, an OTP or "
    "a personal message).",
    LABEL_PROMO: "This looks like marketing from a business: an offer, discount or "
    "advert. It may be unwanted, but it does not look deceptive.",
    LABEL_SCAM: "This looks like a scam or phishing attempt. Do not click links, call "
    "numbers in it or share codes. Contact the organisation through its official app or "
    "website instead.",
}


def sms_tab() -> None:
    try:
        detector = get_sms_detector()
    except DetectorError as exc:
        st.error(str(exc))
        return
    for warning in detector.warnings:
        st.info(warning)

    example_row(
        "sms",
        [
            (
                "Bank alert",
                {
                    "sms_message": "Dear customer, A/C XX1234 debited by Rs 450.00 on 12-10-26 "
                    "at a store. Avl bal Rs 10,200. If not you, call the number on your card."
                },
            ),
            (
                "Store offer",
                {
                    "sms_message": "FLAT 40% OFF on all pizzas this weekend at Pizza Corner! Show "
                    "this SMS at the counter. T&C apply."
                },
            ),
            (
                "Scam",
                {
                    "sms_message": "URGENT: your bank account is suspended. Verify your KYC now at "
                    "http://secure-bank-kyc.example/verify or it will be blocked in 24 hours."
                },
            ),
        ],
    )
    message = st.text_area(
        "SMS message",
        key="sms_message",
        height=140,
        max_chars=MAX_SMS_CHARS + 200,
        placeholder="Paste a text message here...",
        help="Your message is analysed in memory and is not stored.",
    )
    if st.button("Analyze SMS", type="primary", key="sms_analyze") and not too_many_requests():
        try:
            result = detector.predict(message)
        except DetectorError as exc:
            st.error(str(exc))
        else:
            if result.label == LABEL_OK:
                st.success(f"**{result.label}**")
            elif result.label == LABEL_PROMO:
                st.warning(f"**{result.label}**")
            else:
                st.error(f"**{result.label}**")
            level = (
                "ok"
                if result.label == LABEL_OK
                else "warn"
                if result.label == LABEL_PROMO
                else "bad"
            )
            st.html(
                components.result_card(
                    level,
                    result.label,
                    SMS_EXPLANATIONS.get(result.label, "This looks like an unwanted message."),
                    result.score,
                    result.threshold,
                    "unwanted score",
                )
            )
            dashboard.record_activity("SMS", result.label)
            columns = st.columns(2)
            columns[0].metric("Unwanted-message score", f"{result.score:.2f}")
            if result.scam_score is not None:
                columns[1].metric("Scam likelihood", f"{result.scam_score:.2f}")
            st.caption(
                f"The unwanted-message score is the main model's confidence that this is not an "
                f"ordinary message; scores at or above {result.threshold:.2f} are flagged. These "
                "are model scores, not guarantees, and real-world spam is rarer than in the "
                "training data."
            )
            if result.influential_words:
                st.html(
                    components.chips(
                        "Words that pushed the score up", result.influential_words, hot=True
                    )
                )

    with st.expander("About the SMS models"):
        meta = detector.metadata
        test = meta["metrics"]["test"]
        external = meta["metrics"]["external_india"]
        st.write(
            f"Main model: `{meta['model']}` trained {meta['created_utc'][:10]} on "
            f"{meta['data_sizes']['train']:,} messages (data from roughly 2011-2021). "
            "It separates ordinary messages from unwanted ones (promotions and scams)."
        )
        st.write(
            f"On held-out similar data: precision {test['precision']:.2f}, "
            f"recall {test['recall']:.2f}. "
            f"On messages from a different source: precision {external['precision']:.2f}, "
            f"recall {external['recall']:.2f}."
        )
        for item in meta["limitations"]:
            st.write(f"- {item}")
        if detector.type_metadata:
            tmeta = detector.type_metadata
            cv = tmeta["metrics"]["cross_validated_dev"]
            sizes = tmeta["data_sizes"]
            st.write(
                f"Promotional-vs-scam model: `{tmeta['model']}`, trained on only "
                f"{sizes['dev']} hand-labelled messages. Cross-validated accuracy "
                f"{cv['accuracy']:.2f} (scam precision {cv['precision']:.2f}, "
                f"recall {cv['recall']:.2f})."
            )
            for item in tmeta["limitations"]:
                st.write(f"- {item}")


def email_tab() -> None:
    try:
        detector = get_email_detector()
    except DetectorError as exc:
        st.error(str(exc))
        return
    for warning in detector.warnings:
        st.info(warning)
    challenge = detector.metadata.get("metrics", {}).get("challenge_modern")
    legit_total = challenge["fp"] + challenge["tn"] if challenge else 0
    high_false_alarms = bool(legit_total) and challenge["fp"] / legit_total > 0.15
    if challenge and (not detector.metadata.get("guard_passed", True) or high_false_alarms):
        wrongly = challenge["fp"]
        st.warning(
            f"**Known weakness:** in a small test, this model wrongly flagged {wrongly} of "
            f"{legit_total} ordinary modern emails (such as statements, verification codes, "
            "receipts and newsletters), because it was trained on 2000s office email. Treat a "
            "'Spam' result as a reason to look more closely, not as proof."
        )

    example_row(
        "email",
        [
            (
                "Team update",
                {
                    "email_subject": "Agenda for Monday",
                    "email_body": "Hi team, attached is the agenda for Monday's planning meeting. "
                    "Please add your items before Friday. Thanks, Priya",
                },
            ),
            (
                "Phishing",
                {
                    "email_subject": "Action required: verify your account",
                    "email_body": "Dear user, we detected unusual activity. Confirm your password "
                    "within 24 hours at http://account-verify.example/login or your mailbox will "
                    "be closed.",
                },
            ),
            (
                "Prize spam",
                {
                    "email_subject": "You have won a free prize!!!",
                    "email_body": "Congratulations! Click the link now to claim your free cash "
                    "prize before it expires. Limited offer, act now.",
                },
            ),
        ],
    )
    subject = st.text_input(
        "Email subject",
        key="email_subject",
        max_chars=MAX_SUBJECT_CHARS + 100,
        help="Your email is analysed in memory and is not stored. Links are never opened.",
    )
    body = st.text_area(
        "Email body",
        key="email_body",
        height=220,
        max_chars=MAX_BODY_INPUT_CHARS + 500,
        placeholder="Paste the email text here (plain text; pasted HTML is reduced to its text)...",
    )
    if st.button("Analyze email", type="primary", key="email_analyze") and not too_many_requests():
        try:
            result = detector.predict(subject, body)
        except DetectorError as exc:
            st.error(str(exc))
        else:
            if result.is_spam:
                st.error(f"**{result.label}**")
                note = (
                    "This email looks like spam or phishing. Do not click links, open "
                    "attachments or reply. If it claims to be from an organisation, go to "
                    "its official website yourself."
                )
            else:
                st.success(f"**{result.label}**")
                note = (
                    "Nothing in this email looked like the spam the model was trained on. "
                    "That is not a guarantee: new kinds of phishing can look ordinary."
                )
            st.html(
                components.result_card(
                    "bad" if result.is_spam else "ok",
                    result.label,
                    note,
                    result.score,
                    result.threshold,
                    "spam score",
                )
            )
            dashboard.record_activity("Email", result.label)
            st.metric("Spam score", f"{result.score:.2f}")
            st.caption(
                f"Scores at or above {result.threshold:.2f} are flagged. This is a model "
                "score, not a probability for your inbox."
            )
            if result.influential_terms:
                st.html(
                    components.chips(
                        "Terms that pushed the score up", result.influential_terms, hot=True
                    )
                )
            if result.truncated:
                st.caption("Only the first 6,000 characters of the body were analysed.")

    with st.expander("About the email model"):
        meta = detector.metadata
        test = meta["metrics"]["test"]
        ext = meta["external_2026"]
        st.write(
            f"Model: `{meta['model']}` trained {meta['created_utc'][:10]} on "
            f"{meta['data_sizes']['train']:,} emails from the Enron-Spam collection "
            "(legitimate mail from 1999-2002, spam from 2001-2005)."
        )
        st.write(
            f"On held-out data of the same kind: precision {test['precision']:.2f}, "
            f"recall {test['recall']:.2f}. It flagged {ext['detection_rate']:.0%} of "
            f"{ext['n']:,} real spam emails from June-October 2026. That test contains no "
            "legitimate email, so it says nothing about false alarms."
        )
        for item in meta["limitations"]:
            st.write(f"- {item}")


@st.cache_resource(show_spinner=False)
def get_toxicity_detector() -> ToxicityDetector:
    """Load the comment model once per server process."""
    return load_toxicity_detector()


LABEL_MEANING = {
    "toxic": "rude, disrespectful or unreasonable",
    "severe_toxic": "very hateful or aggressive",
    "obscene": "vulgar or profane language",
    "threat": "wishes or intends to hurt someone",
    "insult": "insulting or demeaning a person",
    "identity_hate": "attacks a group because of who they are",
}


def comments_tab() -> None:
    try:
        detector = get_toxicity_detector()
    except DetectorError as exc:
        st.error(str(exc))
        return
    for warning in detector.warnings:
        st.info(warning)
    bias = detector.metadata.get("bias_probe")
    if bias and not bias.get("acceptable", True):
        st.warning(
            "**Known weakness:** in a small test with neutral sentences that mention an identity "
            "(for example 'I am a ... person'), the model flagged up to "
            f"{bias['max_flagged_share']:.0%} of the sentences for some groups (worst: "
            f"'{bias['worst_term']}'). It can mistake the mention of an identity for toxicity."
        )
    st.info(
        "The model reads words, not meaning. It can misjudge quotes, sarcasm, discussions about "
        "toxic language, and sentences that mention an identity. Use it as a prompt for a human "
        "to look, not as a verdict."
    )

    example_row(
        "comment",
        [
            (
                "Friendly",
                {
                    "comment_text": "Thanks for the detailed explanation, this really helped me "
                    "understand the policy."
                },
            ),
            (
                "Insult",
                {
                    "comment_text": "You are a complete idiot and nobody wants your stupid "
                    "edits here."
                },
            ),
            (
                "Neutral identity mention",
                {"comment_text": "I am a gay person and I enjoy hiking in the mountains."},
            ),
        ],
    )
    comment = st.text_area(
        "Comment",
        key="comment_text",
        height=140,
        max_chars=MAX_COMMENT_CHARS + 300,
        placeholder="Paste a comment here...",
        help="Your comment is analysed in memory and is not stored.",
    )
    if (
        st.button("Analyze comment", type="primary", key="comment_analyze")
        and not too_many_requests()
    ):
        try:
            result = detector.predict(comment)
        except DetectorError as exc:
            st.error(str(exc))
        else:
            if result.any_flagged:
                st.warning("**Flagged:** " + ", ".join(result.flagged))
            else:
                st.success("**No label reached its threshold.**")
            top_label = max(TOXICITY_LABELS, key=lambda name: result.scores[name])
            st.html(
                components.result_card(
                    "bad" if result.any_flagged else "ok",
                    "Flagged: " + ", ".join(result.flagged)
                    if result.any_flagged
                    else "Nothing flagged",
                    f"The highest of the six label scores is '{top_label}'. Each label has its "
                    "own flag threshold.",
                    result.scores[top_label],
                    result.thresholds[top_label],
                    "top score",
                )
            )
            dashboard.record_activity(
                "Comments", "Flagged" if result.any_flagged else "Nothing flagged"
            )
            for label in TOXICITY_LABELS:
                score, limit = result.scores[label], result.thresholds[label]
                name_col, bar_col, num_col = st.columns([3, 5, 3])
                flag = " - FLAGGED" if label in result.flagged else ""
                name_col.write(f"**{label}**{flag}")
                bar_col.progress(min(max(score, 0.0), 1.0))
                num_col.caption(f"score {score:.2f} (flag at {limit:.2f})")
            for label, terms in result.top_terms.items():
                if terms:
                    st.write(
                        f"`{label}` ({LABEL_MEANING[label]}) - words pushing it up: "
                        + ", ".join(f"`{t}`" for t in terms)
                    )
            if result.truncated:
                st.caption("Only the first part of a long comment is analysed.")

    with st.expander("About the comment model"):
        meta = detector.metadata
        summary = meta["metrics"]["official_test"]["summary"]
        st.write(
            f"Model: `{meta['model']}` trained {meta['created_utc'][:10]} on "
            f"{meta['data_sizes']['train']:,} Wikipedia talk-page comments (about 2017). On the "
            f"official held-out test set: macro-F1 {summary['macro_f1']:.2f}, "
            f"micro-F1 {summary['micro_f1']:.2f}."
        )
        per_label = meta["metrics"]["official_test"]["per_label"]
        st.table(
            {
                "label": TOXICITY_LABELS,
                "precision": [f"{per_label[label]['precision']:.2f}" for label in TOXICITY_LABELS],
                "recall": [f"{per_label[label]['recall']:.2f}" for label in TOXICITY_LABELS],
                "F1": [f"{per_label[label]['f1']:.2f}" for label in TOXICITY_LABELS],
                "positive examples in test": [
                    per_label[label]["positives"] for label in TOXICITY_LABELS
                ],
            }
        )
        for item in meta["limitations"]:
            st.write(f"- {item}")


@st.cache_resource(show_spinner=False)
def get_phishing_detector() -> PhishingDetector:
    """Load the URL model once per server process."""
    return load_phishing_detector()


FEATURE_LABELS = {
    "url_len": "Address length (characters)",
    "n_subdomains": "Subdomain levels",
    "is_ip_host": "Numeric IP instead of a domain",
    "has_at": "Contains '@'",
    "host_hyphens": "Hyphens in the domain",
    "path_depth": "Path depth",
    "n_login_words_host": "Login-type words in the domain",
    "brand_not_registered_domain": "Brand name that is not the real domain",
    "is_hosting_platform": "Free hosting platform",
    "tld_suspicious": "Throw-away style ending",
}


def urls_tab() -> None:
    try:
        detector = get_phishing_detector()
    except DetectorError as exc:
        st.error(str(exc))
        return
    for warning in detector.warnings:
        st.info(warning)
    st.info(
        "The address is analysed as text only. It is never opened and never shown as a link. "
        "Only the part up to the path is analysed: anything after a '?' or '#' is ignored."
    )
    challenge = detector.metadata.get("metrics", {}).get("challenge")
    if not detector.metadata.get("guard_passed", True) and challenge:
        st.warning(
            f"**Known weakness:** in a small test, this model wrongly flagged "
            f"{challenge['legit_flagged']} of {challenge['legit_n']} ordinary links (including "
            "genuine login pages and pages on free hosting platforms), because public training "
            "data has few legitimate URLs like these. A 'Potentially phishing' result means "
            "'look closely', not 'confirmed'."
        )

    example_row(
        "url",
        [
            ("Familiar site", {"url_input": "https://en.wikipedia.org/wiki/Machine_learning"}),
            (
                "Look-alike login",
                {"url_input": "http://paypal.com.secure-login-verify.example/account/update"},
            ),
            ("Shortened link", {"url_input": "https://bit.ly/3abcdef"}),
        ],
    )
    url = st.text_input(
        "URL to check",
        key="url_input",
        max_chars=MAX_URL_CHARS + 100,
        placeholder="https://example.com/page",
        help="The address is analysed in memory and is not stored. It is never opened.",
    )
    if st.button("Check URL", type="primary", key="url_analyze") and not too_many_requests():
        try:
            result = detector.predict(url)
        except DetectorError as exc:
            st.error(str(exc))
        else:
            st.code(result.url, language=None)  # plain text: never a clickable link
            if not result.judged:
                st.info(f"**{result.label}**")
            elif result.is_phishing:
                st.error(f"**{result.label}**")
            else:
                st.success(f"**{result.label}**")
            level = "warn" if not result.judged else "bad" if result.is_phishing else "ok"
            st.html(
                components.result_card(
                    level,
                    result.label,
                    "Judged from the text of the address alone; the page itself is never opened.",
                    result.score,
                    result.threshold,
                    "model score",
                )
            )
            dashboard.record_activity("URL", result.label)
            st.metric("Model score", f"{result.score:.2f}")
            st.caption(
                f"Scores at or above {result.threshold:.2f} are called potentially phishing. "
                "The score is the model's confidence from the text of the address alone. It "
                "cannot confirm that a site is safe: it does not see the page, the certificate, "
                "the domain's age or any blocklist."
            )
            if result.assumed_scheme:
                st.caption(
                    "No http:// or https:// was typed. The model does not use the scheme, so "
                    "this does not change the result."
                )
            if result.features.get("is_hosting_platform"):
                st.caption(
                    "This page is on a free hosting platform. In the training data almost every "
                    "address on such platforms was phishing, so the score reflects that base rate, "
                    "not evidence about this page: legitimate pages exist there too."
                )
            if result.ignored_query:
                st.caption("Everything after '?' or '#' was ignored.")
            if result.observations:
                st.write("**Things worth noticing in this address:**")
                for note in result.observations:
                    st.write(f"- {note}")
            with st.expander("Features read from the address"):
                st.table(
                    {
                        "feature": [FEATURE_LABELS[k] for k in FEATURE_LABELS],
                        "value": [f"{result.features[k]:g}" for k in FEATURE_LABELS],
                    }
                )

    with st.expander("About the URL model"):
        meta = detector.metadata
        test = meta["metrics"]["test_temporal"]
        st.write(
            f"Model: `{meta['model']}` trained {meta['created_utc'][:10]} on "
            f"{meta['data_sizes']['train']:,} URLs collected between "
            f"{meta['date_ranges']['train+val'].replace(' to ', ' and ')}. On newer URLs "
            f"({meta['date_ranges']['test'].replace(' to ', ' to ')}) it caught "
            f"{test['recall']:.0%} of phishing URLs while flagging "
            f"{test['false_positive_rate']:.1%} "
            "of legitimate ones."
        )
        for item in meta["limitations"]:
            st.write(f"- {item}")


def home_tab() -> None:
    ready = sum(1 for item in summarize_models() if item.available)
    st.html(
        components.hero(
            "Stay one step ahead of scams.",
            "Four small machine-learning models read the text of messages, emails, comments "
            "and web addresses and tell you what looks suspicious, and why.",
            f"{ready} of 4 detectors online",
        )
        + components.feature_cards(
            [
                ("sms", "SMS", "Ordinary, promotional or scam: three classes, not just spam."),
                ("email", "Email", "Spam and phishing, tested on real 2026 spam."),
                ("comment", "Comments", "Six kinds of toxicity, each with its own threshold."),
                ("url", "URLs", "Phishing from the address text alone. Never opened."),
            ]
        )
        + components.section("How a check works")
        + components.pipeline(
            [
                ("1 Input", "text you paste, checked and length-limited"),
                ("2 Clean", "same normalisation as in training"),
                ("3 Features", "word / character patterns or URL measurements"),
                ("4 Model", "a small scikit-learn classifier gives a score"),
                ("5 Threshold", "chosen on validation data, not guessed"),
                ("6 Explain", "the words or features behind the score"),
            ]
        )
    )
    st.markdown(
        "- **SMS**: ordinary, promotional or scam message\n"
        "- **Email**: spam / phishing or not\n"
        "- **Comments**: six kinds of toxicity, each with its own score\n"
        "- **URL**: potentially phishing or likely legitimate (the link is never opened)"
    )
    st.markdown("#### How it works")
    st.write(
        "Each model learned from thousands of labelled examples. It turns your text into "
        "numbers (counts of words and short character sequences, or measurements of an "
        "address) and computes a score between 0 and 1. A score above a threshold chosen on "
        "separate validation data becomes a flag. It does not understand meaning, so it can "
        "be fooled and it can be wrong."
    )
    st.markdown("#### Your privacy")
    st.write(
        "What you paste is analysed in memory and not stored. Links are never opened. "
        "The models run on this machine; nothing is sent to an outside service. The Dashboard "
        "keeps only counts of verdicts for this browser session, never your text."
    )
    st.markdown("#### Which detectors are ready")
    for item in summarize_models():
        if item.available:
            st.write(f"- {item.title}: ready ({item.headline})")
        else:
            st.write(f"- {item.title}: **model not available** (see the About tab)")


def about_tab() -> None:
    st.subheader("About the models")
    st.write(
        "Every figure below is read from the model's own saved results, so it matches the "
        "file actually in use. Scores on the data a model was trained from look better than "
        "real life; the limitations are part of the result."
    )
    for item in summarize_models():
        with st.expander(item.title, expanded=False):
            if not item.available:
                st.error(
                    "This model is not available. Train it with the scripts in "
                    "src/training/ (see the README), then reload the page."
                )
                continue
            st.write(
                f"Model: `{item.model}`, trained {item.trained_on} on "
                f"{item.train_size:,} examples. Result: {item.headline}."
            )
            for limitation in item.limitations:
                st.write(f"- {limitation}")
    st.markdown("#### Responsible use")
    st.write(
        "- A flag is a prompt to look closer, never proof. Do not punish, report or block "
        "anyone because of a score.\n"
        "- The comment model can mistake the mention of an identity for toxicity, so do not "
        "use it to moderate people automatically.\n"
        "- A 'safe' result is not a guarantee: scams change faster than the training data.\n"
        "- Do not paste passwords, card numbers or other secrets. Input is not stored, but "
        "you do not need to share them to get a result."
    )
    st.markdown("#### Limits of this project")
    st.write(
        "This is a learning project, not a security product. It cannot see attachments, "
        "pages behind links, sender reputation or message history. Do not use it to make "
        "decisions about people."
    )


st.title("SmartShield AI")
st.warning(
    "Predictions can be wrong. This is a learning project, not a security product. "
    "Never rely on it to decide whether a message is safe."
)
# on_change="rerun" lets the script know which tab is open, so the heavy Dashboard is only drawn
# when someone is looking at it (otherwise every click anywhere would redraw its charts).
home, sms, email_page, comments, urls, dash, about = st.tabs(
    ["Home", "SMS", "Email", "Comments", "URL", "Dashboard", "About"],
    key="main_tabs",
    on_change="rerun",
)
with home:
    home_tab()
with sms:
    sms_tab()
with email_page:
    email_tab()
with comments:
    comments_tab()
with urls:
    urls_tab()
with dash:
    if dash.open:
        dashboard.render()
with about:
    about_tab()
