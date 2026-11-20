"""Export example sentences behind each result, so correct and wrong predictions can be read and explained.

Writes results/examples/<topic>.csv with the sentence (target in [brackets]), gold and predicted senses and
their WordNet glosses. Run with: python -m wsd_difficulty.examples
"""

import random

import pandas as pd

from . import config
from .pipeline import DISCOURSE_PREDICTIONS_FILE, PREDICTIONS_FILE, load_prepared
from .semcor_csv import read_semcor_csv
from .wordnet_utils import synset_of_sense_key

CANDIDATES_PER_OUTCOME = 40
READABLE_LENGTH = (8, 35)


def gloss(sense_key):
    synset = synset_of_sense_key(sense_key) if isinstance(sense_key, str) else None
    return synset.definition() if synset else ""


def marked_sentence(corpus, instance_id):
    sentence_id, position = instance_id.rsplit(".t", 1)
    tokens = corpus.sentences[sentence_id].tokens
    words = [f"[{token.form}]" if index == int(position) else token.form for index, token in enumerate(tokens)]
    return " ".join(words), len(tokens)


def predictions_for(predictions, model, feature_set):
    rows = predictions[(predictions["evaluation"] == "semcor_cv") & (predictions["model"] == model)
                       & (predictions["feature_set"] == feature_set)]
    return rows.set_index("instance_id")


def describe(row_ids, outcome, topic, context):
    """One row per instance with everything needed to read the example."""
    corpus, word_table, examples_by_id, predicted, mfs_sense = context
    rows = []
    for instance_id, predicted_sense, probability in row_ids:
        example = examples_by_id[instance_id]
        sentence, length = marked_sentence(corpus, instance_id)
        if not READABLE_LENGTH[0] <= length <= READABLE_LENGTH[1]:
            continue
        word = word_table.loc[predicted.loc[instance_id, "target_word"]]
        rows.append({
            "topic": topic, "outcome": outcome, "target_word": word.name, "instance_id": instance_id,
            "entropy_tercile": word["entropy_tercile"], "sense_entropy": round(word["sense_entropy"], 2),
            "mfs_probability": round(word["mfs_probability"], 2), "sentence": sentence,
            "gold_sense": example.sense, "gold_gloss": gloss(example.sense),
            "predicted_sense": predicted_sense, "predicted_gloss": gloss(predicted_sense),
            "predicted_probability": round(probability, 2) if pd.notna(probability) else None,
            "mfs_sense": mfs_sense.get(word.name), "mfs_gloss": gloss(mfs_sense.get(word.name)),
        })
    return rows


def one_per_word(candidates, generator):
    """Shuffle and keep at most one instance per target word, so examples cover many words."""
    candidates = list(candidates)
    generator.shuffle(candidates)
    seen, kept = set(), []
    for candidate in candidates:
        word = candidate[0]
        if word not in seen:
            seen.add(word)
            kept.append(candidate[1:])
        if len(kept) == CANDIDATES_PER_OUTCOME * 2:
            break
    return kept


def pick(frame, mask, generator, sense_column="predicted_sense"):
    chosen = frame[mask]
    return one_per_word(zip(chosen["target_word"], chosen.index, chosen[sense_column],
                            chosen["predicted_probability"]), generator)


def previous_mention_rows(corpus, examples_by_id, sentence_level, discourse_level, generator):
    """One sense per discourse: guess each instance's sense from its nearest earlier mention."""
    rows = []
    for instance_id, example in examples_by_id.items():
        earlier = [mention for mention in example.other_mentions if mention.comes_before]
        if not earlier or instance_id not in sentence_level.index:
            continue
        nearest = max(earlier, key=lambda mention: (mention.sentence_gap, mention.instance_id))
        rows.append((nearest.gold_sense in example.gold_senses,
                     sentence_level.loc[instance_id, "target_word"], instance_id, nearest))
    generator.shuffle(rows)

    exported = []
    for outcome in (True, False):
        seen = set()
        for same_sense, word, instance_id, nearest in rows:
            if same_sense != outcome or word in seen or len(seen) >= CANDIDATES_PER_OUTCOME:
                continue
            sentence, length = marked_sentence(corpus, instance_id)
            previous_sentence, previous_length = marked_sentence(corpus, nearest.instance_id)
            if not (READABLE_LENGTH[0] <= length <= READABLE_LENGTH[1] and previous_length <= READABLE_LENGTH[1]):
                continue
            seen.add(word)
            example = examples_by_id[instance_id]
            exported.append({
                "topic": "one_sense_per_discourse", "outcome": "correct" if outcome else "wrong",
                "target_word": word, "instance_id": instance_id, "sentence_gap": -nearest.sentence_gap,
                "previous_sentence": previous_sentence, "previous_sense": nearest.gold_sense,
                "previous_gloss": gloss(nearest.gold_sense), "sentence": sentence,
                "gold_sense": example.sense, "gold_gloss": gloss(example.sense),
                "sentence_model_correct": bool(sentence_level.loc[instance_id, "correct"]),
                "discourse_model_correct": bool(discourse_level.loc[instance_id, "correct"])
                if instance_id in discourse_level.index else None,
            })
    return exported


def export_examples(seed=config.RANDOM_SEED):
    print("Reading SemCor and predictions")
    corpus = read_semcor_csv(config.SEMCOR_CSV_DIR)
    prepared = load_prepared()
    word_table = prepared["word_table"].set_index("target_word")
    examples_by_id = {example.instance_id: example
                      for examples in prepared["training_examples"].values() for example in examples}
    predictions = pd.concat([pd.read_csv(config.PROCESSED_DIR / PREDICTIONS_FILE),
                             pd.read_csv(config.PROCESSED_DIR / DISCOURSE_PREDICTIONS_FILE)], ignore_index=True)

    sentence_level = predictions_for(predictions, "logistic_regression", "all")
    most_frequent = predictions_for(predictions, "mfs", "none")
    discourse_level = predictions_for(predictions, "logistic_regression", "all+discourse")
    oracle_level = predictions_for(predictions, "logistic_regression", "all+discourse_oracle")
    mfs_sense = most_frequent.groupby("target_word")["predicted_sense"].agg(lambda senses: senses.mode()[0]).to_dict()
    context = (corpus, word_table, examples_by_id, sentence_level, mfs_sense)
    generator = random.Random(seed)

    tercile = sentence_level["target_word"].map(word_table["entropy_tercile"])
    correct = sentence_level["correct"].astype(bool)
    mfs_correct = most_frequent["correct"].astype(bool).reindex(sentence_level.index)
    topics = {
        # RQ1: does entropy predict difficulty? Low-entropy successes vs high-entropy failures and successes
        "entropy_and_difficulty": [
            ("correct", pick(sentence_level, correct & (tercile == "high"), generator)),
            ("correct_low_entropy", pick(sentence_level, correct & (tercile == "low"), generator)),
            ("wrong", pick(sentence_level, ~correct & (tercile == "high"), generator)),
            ("wrong_low_entropy", pick(sentence_level, ~correct & (tercile == "low"), generator)),
        ],
        # RQ2: context fixes the MFS guess, or misleads the model away from it
        "context_and_information": [
            ("correct", pick(sentence_level, correct & ~mfs_correct, generator)),
            ("wrong", pick(sentence_level, ~correct & mfs_correct, generator)),
        ],
        # RQ3: errors that did not choose the MFS, and closely related senses told apart correctly
        "confused_senses": [
            ("correct", pick(sentence_level, correct & ~mfs_correct, generator)),
            ("wrong", pick(sentence_level, ~correct & (sentence_level["predicted_sense"]
                                                        != sentence_level["target_word"].map(mfs_sense)), generator)),
        ],
    }

    # RQ4: discourse features fix or break the sentence-level prediction
    paired = discourse_level.join(sentence_level[["correct"]].rename(columns={"correct": "sentence_correct"}),
                                  how="inner")
    paired = paired.join(oracle_level[["correct"]].rename(columns={"correct": "oracle_correct"}), how="left")
    discourse_correct = paired["correct"].astype(bool)
    sentence_correct = paired["sentence_correct"].astype(bool)
    topics["discourse_features"] = [
        ("correct", pick(paired, discourse_correct & ~sentence_correct, generator)),
        ("wrong", pick(paired, ~discourse_correct & sentence_correct, generator)),
        ("oracle_only_correct", pick(paired, ~discourse_correct & paired["oracle_correct"].fillna(False).astype(bool),
                                     generator)),
    ]

    output = config.RESULTS_DIR / "examples"
    output.mkdir(parents=True, exist_ok=True)
    for topic, groups in topics.items():
        rows = []
        for outcome, candidates in groups:
            rows += describe(candidates, outcome, topic, context)[:CANDIDATES_PER_OUTCOME]
        pd.DataFrame(rows).to_csv(output / f"{topic}.csv", index=False)
        print(f"  wrote {topic}.csv ({len(rows)} examples)")

    discourse_rows = previous_mention_rows(corpus, examples_by_id, sentence_level, discourse_level, generator)
    pd.DataFrame(discourse_rows).to_csv(output / "one_sense_per_discourse.csv", index=False)
    print(f"  wrote one_sense_per_discourse.csv ({len(discourse_rows)} examples)")


if __name__ == "__main__":
    export_examples()
