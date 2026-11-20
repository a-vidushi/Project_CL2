"""How much sense uncertainty each context type removes (RQ2)."""

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .experiments import CROSS_VALIDATION


def held_out_cross_entropy(predictions):
    """Mean -log2 p(gold sense) per word, model and feature set, on held-out SemCor folds.

    Instances whose gold sense never occurred in the training folds are left out,
    since no model trained on those folds can assign them probability.
    """
    usable = predictions[
        (predictions["evaluation"] == CROSS_VALIDATION)
        & predictions["gold_seen_in_training"]
        & predictions["gold_probability"].notna()
    ].copy()
    usable["surprisal"] = -np.log2(usable["gold_probability"].clip(lower=1e-12))
    return (usable.groupby(["target_word", "model", "feature_set"])["surprisal"]
            .mean().rename("cross_entropy").reset_index())


def information_gain_table(predictions, model_name="naive_bayes"):
    """I(S; C | w) estimated as H(S | w) - H(S | w, C), both measured as held-out cross-entropy.

    H(S | w) comes from the smoothed sense prior (the MFS model's distribution).
    """
    cross_entropy = held_out_cross_entropy(predictions)
    prior_entropy = (cross_entropy[cross_entropy["model"] == "mfs"]
                     [["target_word", "cross_entropy"]].rename(columns={"cross_entropy": "prior_cross_entropy"}))
    with_context = (cross_entropy[cross_entropy["model"] == model_name]
                    .rename(columns={"cross_entropy": "conditional_cross_entropy"}))
    table = with_context.merge(prior_entropy, on="target_word")
    table["information_gain"] = table["prior_cross_entropy"] - table["conditional_cross_entropy"]
    return table


def binary_feature_mutual_information(feature_presence, senses):
    """I(S; F) in bits for one binary feature."""
    total = len(senses)
    sense_counts = Counter(senses)
    joint = Counter(zip(feature_presence, senses))
    feature_counts = Counter(feature_presence)

    mutual_information = 0.0
    for (present, sense), count in joint.items():
        p_joint = count / total
        p_feature = feature_counts[present] / total
        p_sense = sense_counts[sense] / total
        mutual_information += p_joint * math.log2(p_joint / (p_feature * p_sense))
    return mutual_information


def binary_entropy(probability):
    if probability <= 0 or probability >= 1:
        return 0.0
    return -(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))


def feature_mutual_information(target_word, examples, feature_types, minimum_count=3):
    """MI of each feature with the sense, plus MI normalised by its ceiling min(H(S), H(F)).

    A rare feature can carry at most H(F) bits, so raw MI favours always-present features
    such as POS tags; the normalised score asks how close a feature comes to its own ceiling.
    """
    senses = [example.sense for example in examples]
    sense_entropy = entropy_of_labels(senses)
    rows = []
    for feature_type in feature_types:
        instances_with_feature = defaultdict(set)
        for position, example in enumerate(examples):
            for feature in example.features[feature_type]:
                instances_with_feature[feature].add(position)
        for feature, positions in instances_with_feature.items():
            if len(positions) < minimum_count:
                continue
            presence = [position in positions for position in range(len(examples))]
            mutual_information = binary_feature_mutual_information(presence, senses)
            ceiling = min(sense_entropy, binary_entropy(len(positions) / len(examples)))
            rows.append({
                "target_word": target_word,
                "feature_type": feature_type,
                "feature": feature,
                "count": len(positions),
                "mutual_information": mutual_information,
                "normalised_mutual_information": mutual_information / ceiling if ceiling > 0 else 0.0,
            })
    return rows


def entropy_of_labels(labels):
    total = len(labels)
    return -sum(count / total * math.log2(count / total) for count in Counter(labels).values())


def build_feature_mutual_information_table(examples_by_word, feature_types):
    rows = []
    for target_word, examples in examples_by_word.items():
        rows += feature_mutual_information(target_word, examples, feature_types)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["rank_within_word"] = table.groupby("target_word")["mutual_information"].rank(ascending=False, method="first")
    return table


def summarise_top_features(mutual_information_table, top_k=10):
    """Per context type: mean raw and normalised MI of its top-k features per word, and its share of each word's top-k."""
    words = mutual_information_table["target_word"].nunique()
    summary = None
    for score in ("mutual_information", "normalised_mutual_information"):
        top = (mutual_information_table.sort_values(score, ascending=False)
               .groupby(["target_word", "feature_type"]).head(top_k))
        per_type = (top.groupby(["target_word", "feature_type"])[score].mean()
                    .groupby("feature_type").mean().rename(f"mean_top{top_k}_{score}"))
        overall_top = (mutual_information_table.sort_values(score, ascending=False)
                       .groupby("target_word").head(top_k))
        share = (overall_top.groupby("feature_type").size() / (words * top_k)).rename(f"share_of_top{top_k}_by_{score}")
        part = pd.concat([per_type, share], axis=1)
        summary = part if summary is None else summary.join(part)
    # Typical feature frequency explains why raw MI favours dense types
    summary["median_feature_count"] = mutual_information_table.groupby("feature_type")["count"].median()
    return summary.fillna(0.0).reset_index()
