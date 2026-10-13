"""Template-written SMS examples that fix a blind spot found after Phase 3.

Problem: legitimate bank/transaction alerts ("Rs 300 credited to your account...") were flagged
as spam, because in the public training data almost every message containing "Rs" or bank words
was a promotion or scam (a source-bias shortcut).

Fix: add examples where the SAME vocabulary (rupee amounts, account numbers, "credited",
"debited", OTP, bills) appears on BOTH sides:
  - legitimate alerts and personal money messages            -> label 0
  - bank / prize / loan / refund phishing and scams           -> label 1
so the model has to look at what the message ASKS (click, call, share OTP) instead of at the
currency. These rows are SYNTHETIC: they are labelled `synthetic_aug`, are added to the
TRAINING split only, and never to validation, test or the external set. Links use the
reserved `.example` domain, so none of them can point anywhere real.
"""

import random

import pandas as pd

from src.data.text_utils import mask_key, normalize_key

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NAMES = ["R. Kumar", "A. Sharma", "S. Iyer", "M. Khan", "P. Das", "L. Fernandes", "T. Singh"]
MERCHANTS = ["Metro Mart", "City Pharmacy", "Fresh Basket", "Fuel Point", "Book Corner", "Taxi App"]
BILLS = ["electricity", "water", "gas", "broadband", "mobile postpaid", "DTH"]
PURPOSES = ["rent", "groceries", "school fees", "EMI", "insurance", "a recharge", "a subscription"]
WORDS = ["secure", "verify", "update", "claim", "reward", "refund", "login", "bank", "account"]


def _money(rng: random.Random) -> str:
    value = rng.choice([rng.randint(10, 999), rng.randint(1000, 99999)])
    number = f"{value:,}" if rng.random() < 0.6 else str(value)
    if rng.random() < 0.3:
        number += ".00" if rng.random() < 0.7 else f".{rng.randint(1, 99):02d}"
    form = rng.choice(["Rs {}", "Rs.{}", "Rs. {}", "INR {}", "₹{}", "Rs {}/-", "Rs. {}/-"])
    return form.format(number)


def _acct(rng: random.Random) -> str:
    last4 = rng.randint(1000, 9999)
    return rng.choice([f"XX{last4}", f"ending {last4}", f"*{last4}", f"A/c XX{last4}"])


def _date(rng: random.Random) -> str:
    day, month, year = rng.randint(1, 28), rng.choice(MONTHS), rng.choice([24, 25, 26])
    return rng.choice([f"{day:02d}-{month}-{year}", f"{day:02d}/{rng.randint(1, 12):02d}/{year}"])


def _ref(rng: random.Random) -> str:
    return str(rng.randint(10**5, 10**10))


def _url(rng: random.Random) -> str:
    host = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}"
    return f"http://{host}.{rng.choice(['example', 'test'])}/{rng.randint(100, 99999)}"


def _toll(rng: random.Random) -> str:
    """Toll-free helpline number as printed in genuine bank alerts."""
    return rng.choice(["1800", "1860"]) + "".join(str(rng.randint(0, 9)) for _ in range(7))


def _callnum(rng: random.Random) -> str:
    """Scams use mobile numbers AND toll-free-looking numbers, so a helpline is not a ham signal."""
    return _toll(rng) if rng.random() < 0.5 else _phone(rng)


def _phone(rng: random.Random) -> str:
    return "9" + "".join(str(rng.randint(0, 9)) for _ in range(9))


LEGIT_TEMPLATES = [
    lambda r: (
        f"{_money(r)} credited to your account {_acct(r)} on {_date(r)}. "
        f"Available balance {_money(r)}."
    ),
    lambda r: (
        f"Your account {_acct(r)} was debited by {_money(r)} on {_date(r)} for "
        f"{r.choice(PURPOSES)}. Balance: {_money(r)}."
    ),
    lambda r: (
        f"{_money(r)} sent to {r.choice(NAMES)} via UPI. Transaction id {_ref(r)}. "
        "Check your balance in the banking app."
    ),
    lambda r: (
        f"Cash withdrawal of {_money(r)} on {_date(r)} from card {_acct(r)}. "
        f"Remaining balance {_money(r)}. If this was not you, contact the branch."
    ),
    lambda r: (
        f"Thank you for paying {_money(r)} towards your {r.choice(BILLS)} bill. "
        f"Receipt number {_ref(r)}."
    ),
    lambda r: (
        f"EMI of {_money(r)} was auto-debited from {_acct(r)}. "
        f"Next instalment is due on {_date(r)}."
    ),
    lambda r: (
        f"{r.randint(100000, 999999)} is the OTP to authorise a payment of {_money(r)} "
        f"at {r.choice(MERCHANTS)}. Valid for {r.randint(3, 15)} minutes. Never share your OTP."
    ),
    lambda r: (
        f"Refund of {_money(r)} for order {_ref(r)} has been processed to your original "
        "payment method."
    ),
    lambda r: (
        f"Your {r.choice(BILLS)} bill of {_money(r)} is due on {_date(r)}. "
        "Pay through the official app or at the service counter."
    ),
    lambda r: f"Salary of {_money(r)} has been credited to {_acct(r)} on {_date(r)}.",
    lambda r: (
        f"Card {_acct(r)} statement: total due {_money(r)}, minimum due {_money(r)}, "
        f"payment date {_date(r)}."
    ),
    lambda r: f"You received {_money(r)} from {r.choice(NAMES)} on UPI. Ref {_ref(r)}.",
    lambda r: (
        f"Cheque {_ref(r)} for {_money(r)} was deposited and will clear in "
        f"{r.randint(1, 3)} working days."
    ),
    lambda r: f"Premium of {_money(r)} received for policy {_ref(r)}. Thank you for your payment.",
    lambda r: f"I have sent you {_money(r)} for {r.choice(PURPOSES)}, please check and confirm.",
    lambda r: (
        f"{_acct(r)} debited {_money(r)} on {_date(r)}. Txn ref {_ref(r)}. "
        f"Not you? Call {_toll(r)} now."
    ),
    lambda r: (
        f"Dear customer, {_money(r)} debited from {_acct(r)} on {_date(r)} via UPI. "
        f"Ref {_ref(r)}. Dispute? Call {_toll(r)}."
    ),
    lambda r: (
        f"Txn alert: {_money(r)} spent on card {_acct(r)} at {r.choice(MERCHANTS)}. "
        f"Avl limit {_money(r)}. If not done by you call {_toll(r)}."
    ),
    lambda r: (
        f"{_money(r)} w/d from {_acct(r)} at ATM on {_date(r)}. Bal {_money(r)}. "
        f"Queries: {_toll(r)}."
    ),
    lambda r: (
        f"Your a/c {_acct(r)} credited {_money(r)} on {_date(r)} by NEFT from "
        f"{r.choice(NAMES)}. Bal {_money(r)}. For help call {_toll(r)}."
    ),
    lambda r: (
        f"Dear user, {_money(r)} paid to {r.choice(MERCHANTS)} from {_acct(r)} on {_date(r)}. "
        f"Ref no {_ref(r)}. Report fraud: {_toll(r)}."
    ),
    lambda r: f"Can you pay {_money(r)} for {r.choice(PURPOSES)} today? I will return it Friday.",
]

SPAM_TEMPLATES = [
    lambda r: (
        f"Dear customer, your account {_acct(r)} is suspended. Verify now at {_url(r)} "
        "to avoid permanent closure."
    ),
    lambda r: (
        f"URGENT! Your KYC is incomplete. Update within {r.randint(2, 48)} hours at "
        f"{_url(r)} or your account will be blocked."
    ),
    lambda r: (
        f"Congratulations! You have won {_money(r)} in our lucky draw. Call {_callnum(r)} "
        "now to claim your prize."
    ),
    lambda r: f"Get an instant loan of up to {_money(r)} with no documents. Apply now: {_url(r)}",
    lambda r: f"Your reward points worth {_money(r)} expire today. Redeem immediately at {_url(r)}",
    lambda r: (
        f"To receive your refund of {_money(r)}, reply with the OTP you just received "
        f"or call {_callnum(r)}."
    ),
    lambda r: f"Earn {_money(r)} daily from home! No experience needed. Join now: {_url(r)}",
    lambda r: (
        f"Your electricity will be disconnected tonight unless you pay {_money(r)} now. "
        f"Call {_callnum(r)} immediately."
    ),
    lambda r: (
        f"Win a free phone worth {_money(r)}! Text WIN to {r.randint(50000, 89999)} now. "
        "T&Cs apply."
    ),
    lambda r: (
        f"Your card {_acct(r)} is blocked. Click {_url(r)} to reactivate and confirm your PIN."
    ),
    lambda r: f"Claim your income tax refund of {_money(r)} now: {_url(r)}",
    lambda r: (
        f"Recharge with {_money(r)} and get unlimited calls for a year!! Offer ends today {_url(r)}"
    ),
    lambda r: (
        f"{_money(r)} has been credited to your account by mistake. Return it today at "
        f"{_url(r)} or face legal action."
    ),
    lambda r: (
        f"Your {r.choice(BILLS)} connection will be cut. Pay {_money(r)} at {_url(r)} "
        "within 1 hour."
    ),
]


def generate_augmentation(n_each: int, seed: int) -> pd.DataFrame:
    """Return `n_each` unique legitimate and `n_each` unique scam messages (label 0 / 1)."""
    rng = random.Random(seed)
    rows, seen = [], set()
    for label, templates in ((0, LEGIT_TEMPLATES), (1, SPAM_TEMPLATES)):
        made, attempts = 0, 0
        while made < n_each and attempts < n_each * 50:
            attempts += 1
            text = templates[made % len(templates)](rng)
            key = normalize_key(text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label})
            made += 1
    df = pd.DataFrame(rows)
    df["source"] = "synthetic_aug"
    df["original_label"] = df["label"].map({0: "synthetic_legit", 1: "synthetic_scam"})
    df["group"] = df["text"].map(mask_key)
    return df
