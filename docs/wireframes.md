# SmartShield AI: Page Wireframes (text sketches)

Wireframes show **what is on each page, not how it looks**. Colours and styling come later (Phase 8).

## Home

```text
+--------------------------------------------------------------+
| SmartShield AI                                               |
| Spam, phishing and harmful-content detection (learning demo) |
+--------------------------------------------------------------+
| [!] Predictions can be wrong. Never rely on this alone.      |
+--------------------------------------------------------------+
|  [ Spam ]        [ Phishing URL ]       [ Harmful content ]  |
|  Email / SMS     Check a link           Check a comment      |
+--------------------------------------------------------------+
| How it works:  your text -> features -> trained model -> score|
| Nothing you type is stored.                                  |
+--------------------------------------------------------------+
```

## Spam page

```text
Type:  (o) Email   ( ) SMS
Subject: [____________________]      <- email only
Message: [                    ]
         [                    ]
                [ Analyze ]
--------------------------------------------------------------
Result: SPAM  /  NOT SPAM         Model score: 0.93
Why: words that raised the score: "winner", "claim", "£"
[!] Trained on <dataset, year>. Modern spam may differ.
```

## Phishing page

```text
URL: [__________________________________]   [ Analyze ]
(URL is analysed as text only. It is never opened.)
--------------------------------------------------------------
Result: POTENTIALLY PHISHING / LIKELY LEGITIMATE   Score: 0.78
Features checked: length 87 | subdomains 3 | has '@' no | IP host no
[!] This cannot confirm a site is safe.
```

## Harmful-content page

```text
Comment: [                                  ]   [ Analyze ]
--------------------------------------------------------------
toxic          ########..  0.81   FLAGGED
severe_toxic   #.........  0.07
obscene        #####.....  0.52   FLAGGED
threat         ..........  0.02
insult         ######....  0.63   FLAGGED
identity_hate  #.........  0.05
[!] May misread context, quotes, sarcasm and identity terms.
```

## About the models

```text
Per model: dataset + year | split sizes | metrics (in-domain and
modern external test) | threshold | known limitations
(all values read from models/<module>/metadata.json)
```

## Error states (every page)

- Empty input: "Please enter some text."
- Too long: "Input is limited to N characters."
- Invalid URL: "That does not look like a valid URL."
- Missing model: "The <module> model is not available. Run the training script first."
