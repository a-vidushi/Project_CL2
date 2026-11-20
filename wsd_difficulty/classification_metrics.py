"""Precision, recall, F1 and confusion matrices for the sense classifiers.

Each target word is its own multi-class problem, so scores are computed per word (macro-averaged over
that word's senses) and then averaged over words. Accuracy equals micro-averaged precision, recall
and F1 here, because every instance gets exactly one predicted sense.
"""

import pandas as pd

from . import config
from .experiments import CROSS_VALIDATION
from .wordnet_utils import synset_of_sense_key

SHOWCASE_WORDS = ("serve.VERB", "find.VERB", "position.NOUN", "head.NOUN")
MINIMUM_SENSE_COUNT = 5


def scored_rows(predictions):
    """Held-out SemCor rows with a single gold label: the predicted sense if it was one of the gold senses."""
    rows = predictions[predictions["evaluation"] == CROSS_VALIDATION].copy()
    first_gold = rows["gold_senses"].str.split("|").str[0]
    rows["gold_label"] = first_gold.where(~rows["correct"].astype(bool), rows["predicted_sense"])
    return rows


def per_sense_scores(gold, predicted):
    """Precision, recall and F1 of every sense that occurs as gold or prediction."""
    rows = []
    for sense in sorted(set(gold) | set(predicted)):
        true_positives = ((gold == sense) & (predicted == sense)).sum()
        predicted_count, gold_count = (predicted == sense).sum(), (gold == sense).sum()
        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"sense": sense, "gold_count": gold_count, "predicted_count": predicted_count,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def word_scores(predictions):
    """Macro precision, recall and F1 over the gold senses of each word, per model and feature set."""
    rows = []
    for (model, feature_set, word), group in scored_rows(predictions).groupby(
            ["model", "feature_set", "target_word"]):
        senses = per_sense_scores(group["gold_label"].to_numpy(), group["predicted_sense"].to_numpy())
        # Average over senses that actually occur as gold, so a never-predicted rare sense counts as 0
        gold_senses = senses[senses["gold_count"] > 0]
        rows.append({"model": model, "feature_set": feature_set, "target_word": word,
                     "instances": len(group), "accuracy": group["correct"].astype(bool).mean(),
                     "macro_precision": gold_senses["precision"].mean(),
                     "macro_recall": gold_senses["recall"].mean(),
                     "macro_f1": gold_senses["f1"].mean()})
    return pd.DataFrame(rows)


def summarise_word_scores(scores, word_table):
    """Mean of the per-word scores, overall and by entropy tercile."""
    merged = scores.merge(word_table[["target_word", "entropy_tercile"]], on="target_word")
    merged = pd.concat([merged, merged.assign(entropy_tercile="ALL")])
    columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary = merged.groupby(["model", "feature_set", "entropy_tercile"], observed=True)[columns].mean()
    summary.insert(0, "target_words", merged.groupby(["model", "feature_set", "entropy_tercile"],
                                                     observed=True).size())
    return summary.reset_index()


def sense_label(sense_key, width=40):
    synset = synset_of_sense_key(sense_key)
    gloss = synset.definition() if synset else ""
    short = gloss if len(gloss) <= width else gloss[:width - 1] + "…"
    return f"{sense_key} ({short})"


def confusion_matrix(predictions, target_word, model="logistic_regression", feature_set="all"):
    """Gold senses as rows, predicted senses as columns; senses rarer than 5 gold instances become 'other'."""
    rows = scored_rows(predictions)
    rows = rows[(rows["target_word"] == target_word) & (rows["model"] == model)
                & (rows["feature_set"] == feature_set)]
    frequent = rows["gold_label"].value_counts()
    frequent = set(frequent[frequent >= MINIMUM_SENSE_COUNT].index)

    def label(sense):
        return sense_label(sense) if sense in frequent else "other"

    matrix = pd.crosstab(rows["gold_label"].map(label), rows["predicted_sense"].map(label),
                         rownames=["gold"], colnames=["predicted"])
    order = [sense_label(sense) for sense in rows["gold_label"].value_counts().index if sense in frequent]
    order += ["other"] if "other" in set(matrix.index) | set(matrix.columns) else []
    return matrix.reindex(index=order, columns=order, fill_value=0)


def write_metrics(predictions, word_table, results_dir=config.RESULTS_DIR, save=None):
    def default_save(table, name, index=False):
        table.to_csv(results_dir / name, index=index)
        print(f"  wrote {name}")
    save = save or default_save

    scores = word_scores(predictions)
    save(scores, "precision_recall_f1_by_word.csv")
    save(summarise_word_scores(scores, word_table), "precision_recall_f1.csv")
    for word in SHOWCASE_WORDS:
        if (predictions["target_word"] == word).any():
            save(confusion_matrix(predictions, word), f"confusion_matrix_{word.replace('.', '_')}.csv", index=True)


if __name__ == "__main__":
    from .pipeline import load_all_predictions, load_prepared
    write_metrics(load_all_predictions(), load_prepared()["word_table"])
