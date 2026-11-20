"""Per-word sense distribution statistics: entropy, skew, information content (W1 tables)."""

import math
from collections import Counter

import numpy as np
import pandas as pd

from .wordnet_utils import wordnet_polysemy


def entropy_in_bits(counts):
    total = sum(counts)
    probabilities = [count / total for count in counts if count > 0]
    return -sum(p * math.log2(p) for p in probabilities)


def describe_target_word(target_word, instances):
    lemma, pos = instances[0].lemma, instances[0].pos
    sense_counts = Counter(instance.sense for instance in instances)
    total = sum(sense_counts.values())
    counts = list(sense_counts.values())
    sense_entropy = entropy_in_bits(counts)
    most_frequent_sense, most_frequent_count = sense_counts.most_common(1)[0]
    surprisals = [-math.log2(count / total) for count in counts]

    return {
        "target_word": target_word,
        "lemma": lemma,
        "pos": pos,
        "frequency": total,
        "log_frequency": math.log(total),
        "attested_senses": len(sense_counts),
        "wordnet_senses": wordnet_polysemy(lemma, pos),
        "sense_entropy": sense_entropy,
        # Entropy divided by its maximum for this many attested senses
        "normalised_entropy": sense_entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0,
        "most_frequent_sense": most_frequent_sense,
        "mfs_probability": most_frequent_count / total,
        # Mean surprisal of the attested senses, unweighted (rare senses count as much as common ones)
        "mean_sense_information_content": float(np.mean(surprisals)),
        "max_sense_information_content": max(surprisals),
    }


def assign_entropy_terciles(word_table):
    word_table = word_table.copy()
    word_table["entropy_tercile"] = pd.qcut(
        word_table["sense_entropy"].rank(method="first"), 3, labels=["low", "mid", "high"]
    )
    return word_table


def build_word_table(instances_by_word, target_words):
    rows = [describe_target_word(word, instances_by_word[word]) for word in target_words]
    return assign_entropy_terciles(pd.DataFrame(rows))


def summarise_by_group(word_table, group_columns):
    columns = ["frequency", "attested_senses", "wordnet_senses", "sense_entropy",
               "normalised_entropy", "mfs_probability", "mean_sense_information_content"]
    summary = word_table.groupby(group_columns, observed=True)[columns].mean()
    summary.insert(0, "target_words", word_table.groupby(group_columns, observed=True).size())
    return summary.reset_index()
