"""Document-level sense behaviour (one sense per discourse) and the effect of discourse features (RQ4)."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import config, statistics
from .analysis import stratified_means, word_accuracy_table
from .discourse import sentence_gap_bucket
from .information import held_out_cross_entropy
from .sense_statistics import entropy_in_bits

SENTENCE_LEVEL = "all"


def document_consistency_by_word(examples_by_word):
    """Pairwise sense agreement within documents vs. chance, and H(S|w) vs. H(S|w,d)."""
    rows = []
    for word, examples in examples_by_word.items():
        corpus_counts = Counter(example.sense for example in examples)
        total = sum(corpus_counts.values())
        senses_by_document = defaultdict(list)
        for example in examples:
            senses_by_document[example.document_id].append(example.sense)

        agreeing_pairs = all_pairs = 0
        repeated_mentions = 0
        weighted_document_entropy = 0.0
        for senses in senses_by_document.values():
            if len(senses) < 2:
                continue
            counts = Counter(senses)
            agreeing_pairs += sum(count * (count - 1) / 2 for count in counts.values())
            all_pairs += len(senses) * (len(senses) - 1) / 2
            repeated_mentions += len(senses)
            weighted_document_entropy += len(senses) * entropy_in_bits(list(counts.values()))

        if not all_pairs:
            continue
        rows.append({
            "target_word": word,
            "documents_with_repeats": sum(len(s) >= 2 for s in senses_by_document.values()),
            "mentions_in_repeating_documents": repeated_mentions,
            "pairwise_agreement": agreeing_pairs / all_pairs,
            # Two random mentions drawn from the corpus distribution agree with probability sum p^2
            "chance_agreement": sum((count / total) ** 2 for count in corpus_counts.values()),
            "corpus_entropy": entropy_in_bits(list(corpus_counts.values())),
            "document_entropy": weighted_document_entropy / repeated_mentions,
        })
    table = pd.DataFrame(rows)
    table["agreement_above_chance"] = table["pairwise_agreement"] - table["chance_agreement"]
    table["entropy_reduction"] = table["corpus_entropy"] - table["document_entropy"]
    return table


def summarise_consistency(consistency_by_word, word_table):
    merged = consistency_by_word.merge(word_table[["target_word", "pos", "entropy_tercile"]], on="target_word")
    rows = []
    for grouping in (["pos"], ["entropy_tercile"], ["entropy_tercile", "pos"]):
        for keys, group in merged.groupby(grouping, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            mean_gap, lower, upper = statistics.bootstrap_mean_interval(group["agreement_above_chance"])
            rows.append({
                "pos": dict(zip(grouping, keys)).get("pos", "ALL"),
                "entropy_tercile": dict(zip(grouping, keys)).get("entropy_tercile", "ALL"),
                "target_words": len(group),
                "pairwise_agreement": group["pairwise_agreement"].mean(),
                "chance_agreement": group["chance_agreement"].mean(),
                "agreement_above_chance": mean_gap, "lower": lower, "upper": upper,
                "corpus_entropy": group["corpus_entropy"].mean(),
                "document_entropy": group["document_entropy"].mean(),
            })
    return pd.DataFrame(rows)


def previous_mention_agreement(instances, word_table):
    """How often the nearest previous mention has the same gold sense, by POS and distance."""
    with_previous = instances[instances["has_previous_mention"] == True].merge(  # noqa: E712
        word_table[["target_word", "pos"]], on="target_word")
    with_previous = with_previous.assign(
        distance=with_previous["previous_mention_gap"].map(sentence_gap_bucket),
        same_sense=with_previous["previous_mention_same_sense"].astype(float))
    table = pd.concat([with_previous, with_previous.assign(pos="ALL"), with_previous.assign(distance="ALL"),
                       with_previous.assign(pos="ALL", distance="ALL")])
    return (table.groupby(["pos", "distance"])["same_sense"].agg(instances="size", same_sense_rate="mean")
            .reset_index())


def discourse_gain_by_stratum(predictions, word_table):
    """Per-word accuracy gain of each discourse feature set over the sentence-level model."""
    word_accuracy = word_accuracy_table(predictions)
    sentence_level = (word_accuracy[word_accuracy["feature_set"] == SENTENCE_LEVEL]
                      [["target_word", "model", "accuracy"]].rename(columns={"accuracy": "sentence_accuracy"}))
    discourse = word_accuracy[word_accuracy["feature_set"].str.startswith("all+")].merge(
        sentence_level, on=["target_word", "model"])
    discourse["discourse_gain"] = discourse["accuracy"] - discourse["sentence_accuracy"]
    merged = discourse.merge(word_table[["target_word", "pos", "entropy_tercile"]], on="target_word")
    with_all_terciles = pd.concat([merged, merged.assign(entropy_tercile="ALL")])
    return stratified_means(with_all_terciles, "discourse_gain", ["model", "feature_set", "entropy_tercile", "pos"])


def discourse_significance(predictions):
    rows = []
    for model in config.DISCOURSE_MODELS:
        model_rows = predictions[predictions["model"] == model]
        baseline = model_rows[model_rows["feature_set"] == SENTENCE_LEVEL].set_index("instance_id")["correct"]
        for feature_set, group in model_rows[model_rows["feature_set"].str.startswith("all+")].groupby("feature_set"):
            paired = group.set_index("instance_id")["correct"].to_frame("discourse").join(
                baseline.rename("sentence"), how="inner")
            test = statistics.mcnemar_test(paired["discourse"], paired["sentence"])
            rows.append({"model": model, "feature_set": feature_set, "instances": len(paired),
                         "sentence_accuracy": paired["sentence"].mean(),
                         "discourse_accuracy": paired["discourse"].mean(),
                         "gain": paired["discourse"].mean() - paired["sentence"].mean(),
                         "only_discourse_correct": test["only_first_correct"],
                         "only_sentence_correct": test["only_second_correct"],
                         "mcnemar_p": test["p_value"]})
    return pd.DataFrame(rows)


def discourse_information_gain(predictions, word_table):
    """Bits removed by discourse features beyond the sentence: H(S|w,C) - H(S|w,C,D), held out."""
    cross_entropy = held_out_cross_entropy(predictions)
    sentence_level = (cross_entropy[cross_entropy["feature_set"] == SENTENCE_LEVEL]
                      [["target_word", "model", "cross_entropy"]]
                      .rename(columns={"cross_entropy": "sentence_cross_entropy"}))
    discourse = cross_entropy[cross_entropy["feature_set"].str.startswith("all+")].merge(
        sentence_level, on=["target_word", "model"])
    discourse["discourse_information_gain"] = discourse["sentence_cross_entropy"] - discourse["cross_entropy"]
    merged = discourse.merge(word_table[["target_word", "pos", "entropy_tercile"]], on="target_word")
    with_all_terciles = pd.concat([merged, merged.assign(entropy_tercile="ALL")])
    return stratified_means(with_all_terciles, "discourse_information_gain",
                            ["model", "feature_set", "entropy_tercile", "pos"])


def gain_by_mention_status(predictions, instances, model="logistic_regression"):
    """Accuracy of sentence vs. discourse models for instances with and without an earlier mention."""
    model_rows = predictions[(predictions["model"] == model) & (predictions["evaluation"] == "semcor_cv")]
    wide = model_rows.pivot_table(index="instance_id", columns="feature_set", values="correct", aggfunc="first")
    wide = wide.join(instances.set_index("instance_id")[["has_previous_mention", "mentions_in_document"]])
    wide["mentions"] = pd.cut(wide["mentions_in_document"], [0, 1, 3, 10, np.inf], labels=["1", "2-3", "4-10", "11+"])
    discourse_sets = [column for column in wide.columns if str(column).startswith("all+")]
    # Subsampled words have sentence-level rows for instances the discourse models never saw
    wide = wide.dropna(subset=discourse_sets)

    rows = []
    for (has_previous, mentions), group in wide.groupby(["has_previous_mention", "mentions"], observed=True):
        row = {"has_previous_mention": has_previous, "mentions_in_document": mentions, "instances": len(group),
               "sentence_accuracy": group[SENTENCE_LEVEL].mean()}
        for feature_set in discourse_sets:
            row[f"gain_{feature_set}"] = group[feature_set].mean() - group[SENTENCE_LEVEL].mean()
        rows.append(row)
    return pd.DataFrame(rows)
