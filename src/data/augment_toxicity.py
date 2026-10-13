"""Neutral sentences that mention identity terms, used to reduce identity bias (optional).

Problem (measured in the data report): comments that mention some identity words are labelled
toxic far more often than average (e.g. "gay" 58% vs a 10% base rate), so a model learns "the word
itself is a toxicity signal". A harmless sentence such as "I am a proud gay man" can then be
flagged.

Fix being tested: add harmless sentences where the same identity words appear, labelled clean.
To keep the test honest, TRAINING uses one family of templates and identity terms, while the bias
PROBE (tests/data/toxicity_bias_probe_*) uses different templates AND identity terms never
seen here. All rows are synthetic and used for training only.
"""

import random

import pandas as pd

from config import settings

# Identity terms used in training augmentation
TRAIN_IDENTITIES = [
    "gay", "lesbian", "transgender", "straight", "Muslim", "Christian", "Jewish", "Hindu",
    "atheist", "black", "white", "Asian", "Indian", "American", "Mexican", "blind", "deaf",
    "disabled", "old", "young", "Catholic", "immigrant", "religious", "female", "male",
]  # fmt: skip

# Identity terms held out for the bias probe only (never appear in augmentation)
PROBE_ONLY_IDENTITIES = [
    "bisexual", "Sikh", "Buddhist", "Latino", "Arab", "African", "European", "elderly",
    "pregnant", "autistic", "wheelchair-using", "Native American",
]  # fmt: skip

# Training sentences are built from sentence openers x harmless clauses (about 300 distinct
# templates), so each identity term appears in many different contexts instead of a handful.
TRAIN_FRAMES = [
    "I am a {x} person who",
    "My {x} neighbour",
    "Our {x} colleague",
    "A {x} student in my class",
    "The {x} family next door",
    "My {x} friend",
    "That {x} teacher",
    "A {x} woman I met yesterday",
    "A {x} man from our town",
    "The {x} doctor at the clinic",
    "She is {x} and",
    "He is {x} and",
]
TRAIN_CLAUSES = [
    "loves to cook for everyone.",
    "works at the local library.",
    "enjoys long walks by the river.",
    "volunteers at the food bank every weekend.",
    "plays football on Sundays.",
    "just finished reading a long novel.",
    "teaches maths at the primary school.",
    "is planning a trip to the mountains.",
    "helped me move into my new flat.",
    "writes poems in the evenings.",
    "runs a small bakery downtown.",
    "is learning to play the guitar.",
    "visited the museum with us.",
    "organised the neighbourhood clean-up.",
    "is training for a marathon.",
    "gave a great talk at the conference.",
    "fixed my bicycle for free.",
    "is studying biology at university.",
    "always greets everyone warmly.",
    "bakes the best bread in town.",
    "coaches the children's chess club.",
    "likes gardening and growing tomatoes.",
    "brought lunch for the whole team.",
    "is looking for a new apartment.",
    "has a very funny sense of humour.",
]
TRAIN_TEMPLATES = [f"{frame} {clause}" for frame in TRAIN_FRAMES for clause in TRAIN_CLAUSES]

# Round 4: sentence SHAPES the first family lacks (short statements, questions, praise, quotes of
# lived experience). They never reuse the probe wordings below, and the probe identities that are
# held out still never appear in training.
TRAIN_TEMPLATES_B = [
    "Proud to be {x}.",
    "Being {x} is only one part of who I am.",
    "Is it hard to be {x} in a small town?",
    "{X} folks are welcome in our club.",
    "I have many {x} friends and I love them dearly.",
    "My cousin, who is {x}, just got a new job.",
    "We invited our {x} neighbours to dinner.",
    "As a {x} voter I read the manifesto carefully.",
    "The {x} community organised a charity run.",
    "Thanks to my {x} teacher for the great advice.",
    "She told me she is {x} and I congratulated her on the promotion.",
    "He mentioned that he is {x}, and we talked about football.",
    "Which films by {x} directors would you recommend?",
    "Our new manager is {x} and she is very organised.",
    "{X} history month starts tomorrow at the library.",
    "I read an interview with a {x} author yesterday.",
    "The {x} players won the match.",
    "Why do some people ask me about being {x}? I just like pizza.",
    "Please be kind to {x} people, they are our friends.",
    "We hired a {x} engineer and she fixed the bug.",
    "Congratulations to our {x} colleague on the award!",
    "I am {x}, and I enjoy gardening.",
    "Many {x} families live on our street.",
    "A {x} nurse looked after my grandmother.",
    "There is nothing wrong with being {x}.",
]

# Different wording, used only by the bias probe
PROBE_TEMPLATES = [
    "I am a {x} person.",
    "My colleague is {x} and very kind.",
    "{X} people deserve respect.",
    "Can you recommend a good novel by a {x} writer?",
    "She is happy to be {x}.",
    "The article about {x} history was well written.",
]

CONTROL_WORDS = [
    "tall",
    "short",
    "left-handed",
    "retired",
    "vegetarian",
    "musician",
    "teacher",
    "engineer",
]


def generate_identity_augmentation(
    per_identity: int, seed: int, include_short_forms: bool = False
) -> pd.DataFrame:
    """Clean (all-zero label) synthetic comments mentioning training identity terms.

    Each identity gets `per_identity` DISTINCT sentences (at most len(TRAIN_TEMPLATES)).
    """
    rng = random.Random(seed)
    per_identity = min(per_identity, len(TRAIN_TEMPLATES))
    rows = []
    for identity in TRAIN_IDENTITIES:
        for template in rng.sample(TRAIN_TEMPLATES, per_identity):
            text = template.format(x=identity)
            rows.append({"comment_text": text, **{label: 0 for label in settings.TOXICITY_LABELS}})
        if include_short_forms:
            for template in TRAIN_TEMPLATES_B:
                text = template.format(x=identity, X=identity[0].upper() + identity[1:])
                rows.append(
                    {"comment_text": text, **{label: 0 for label in settings.TOXICITY_LABELS}}
                )
    frame = pd.DataFrame(rows)
    frame["id"] = [f"aug{i:05d}" for i in range(len(frame))]
    frame["group"] = frame["comment_text"].str.lower()
    return frame
