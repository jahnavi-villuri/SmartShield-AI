# Labelling guide: promotional vs scam SMS

Used to hand-label a sample of the messages that the main model already calls "unwanted". The dataset's own sub-category labels were found to be unreliable (a bank phishing link was tagged "telemarketing"; ordinary cashback promos were tagged "UPI fraud"), so they are not used.

**Who labelled:** the AI assistant, following these rules. The labels have **not** been verified by a human. Spot-check a sample before trusting any number built on them.

## Categories

**S = Scam / phishing.** The message tries to deceive the reader or take something from them.
- Pretends to be a bank, government, telecom, courier or utility and asks you to verify, update KYC/PAN, share an OTP/PIN/password, or click a link to avoid blocking, closure or disconnection.
- Claims you have **won** a prize, lottery or gift you never entered, or asks for a fee to release it.
- Advance-fee, fake loans, fake jobs ("earn Rs X daily"), fake refunds, fake parcel/customs fees, "return the money sent by mistake".
- Impersonation or romance/relationship baiting, premium-rate "call now to claim" traps.
- Links that look like look-alike domains, or a request for money or credentials from a stranger.

**P = Promotional.** A business markets its own product or service. Unsolicited, possibly annoying, but not deceptive.
- Discounts, cashback, coupons, sales, recharge/data/plan offers, new-product or event announcements, subscription offers, ringtone/caller-tune services, dating or chat-line adverts, contests that are plainly marketing ("text WIN to enter").
- Real-brand or real-service offers (restaurants, apps, telecoms, retailers, real estate, education, clinics).

**U = Unclear / not usable.** Cannot decide, meaningless text, personal chat mislabelled as spam, a truncated fragment, or a message that is both. These rows are excluded from training and evaluation.

## Tie-break rules
1. If it asks for **credentials, an OTP, or a payment to a stranger**, or says you **won** something you didn't enter: **S**.
2. If it only advertises a discount/offer/service from a named business, even aggressively: **P**.
3. "Win a prize, text WIN to 5xxxx" competition adverts: **P** if framed as entering a competition, **S** if it says you have already won.
4. Still unsure: **U**.

## Known weaknesses of these labels
One labeller, no second opinion, written rules, judgement calls on contests and premium-rate services. Expect some label noise, especially between P and S for prize and premium-rate messages.
