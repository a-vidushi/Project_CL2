"""How related a word's senses are: WordNet path similarity (nouns, verbs) or extended gloss overlap (adjectives, adverbs)."""

import itertools
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

from .wordnet_utils import synset_of_sense_key


def gloss_words(synset):
    return re.findall(r"[a-z]+", synset.definition().lower())


def related_synsets(synset):
    related = set(synset.hypernyms() + synset.hyponyms() + synset.similar_tos() + synset.also_sees() + synset.attributes())
    for lemma in synset.lemmas():
        for linked_lemma in lemma.pertainyms() + lemma.derivationally_related_forms():
            related.add(linked_lemma.synset())
    related.discard(synset)
    return sorted(related, key=lambda s: s.name())


@lru_cache(maxsize=None)
def extended_gloss(synset):
    words = gloss_words(synset)
    for example in synset.examples():
        words += re.findall(r"[a-z]+", example.lower())
    for neighbour in related_synsets(synset):
        words += ["<break>"] + gloss_words(neighbour)
    return tuple(words)


def phrase_overlap_score(first_words, second_words):
    """Banerjee & Pedersen (2003): sum of squared lengths of shared phrases, longest first.

    Phrases made only of stop words do not count.
    """
    first, second = list(first_words), list(second_words)
    score = 0
    while True:
        match = SequenceMatcher(None, first, second, autojunk=False).find_longest_match(0, len(first), 0, len(second))
        if match.size == 0:
            break
        phrase = first[match.a:match.a + match.size]
        if all(word in STOP_WORDS or word == "<break>" for word in phrase):
            break
        score += match.size ** 2
        # Replace the matched phrase with markers that can never match again
        first[match.a:match.a + match.size] = [f"<used-a-{len(first)}-{i}>" for i in range(match.size)]
        second[match.b:match.b + match.size] = [f"<used-b-{len(second)}-{i}>" for i in range(match.size)]
    return score


@lru_cache(maxsize=None)
def sense_pair_relatedness(first_key, second_key, pos):
    first, second = synset_of_sense_key(first_key), synset_of_sense_key(second_key)
    if first is None or second is None:
        return np.nan
    if pos in ("NOUN", "VERB"):
        return first.path_similarity(second) or 0.0
    return float(phrase_overlap_score(extended_gloss(first), extended_gloss(second)))


def describe_word_relatedness(target_word, instances):
    pos = instances[0].pos
    sense_counts = Counter(instance.sense for instance in instances)
    senses = [sense for sense, _ in sense_counts.most_common()]

    pair_scores, pair_weights = [], []
    for first, second in itertools.combinations(senses, 2):
        score = sense_pair_relatedness(first, second, pos)
        if not np.isnan(score):
            pair_scores.append(score)
            pair_weights.append(sense_counts[first] * sense_counts[second])

    top_two = sense_pair_relatedness(senses[0], senses[1], pos) if len(senses) > 1 else np.nan
    return {
        "target_word": target_word,
        "relatedness_measure": "path_similarity" if pos in ("NOUN", "VERB") else "extended_gloss_overlap",
        "mean_pairwise_relatedness": float(np.mean(pair_scores)) if pair_scores else np.nan,
        # Pairs of frequent senses dominate: these are the confusions a classifier actually faces
        "frequency_weighted_relatedness": float(np.average(pair_scores, weights=pair_weights)) if pair_scores else np.nan,
        "top_two_sense_relatedness": top_two,
    }


def confusion_relatedness(predictions, word_table, examples_by_word,
                          model="logistic_regression", feature_set="all"):
    """For every error: is the wrongly chosen sense more related to the gold sense than the other wrong senses?"""
    errors = predictions[(predictions["evaluation"] == "semcor_cv") & (predictions["model"] == model)
                         & (predictions["feature_set"] == feature_set) & ~predictions["correct"]]
    pos_of_word = dict(zip(word_table["target_word"], word_table["pos"]))
    mfs_of_word = dict(zip(word_table["target_word"], word_table["most_frequent_sense"]))
    attested = {word: sorted({example.sense for example in examples}) for word, examples in examples_by_word.items()}

    rows = []
    for word, gold_field, predicted in zip(errors["target_word"], errors["gold_senses"], errors["predicted_sense"]):
        gold_senses = gold_field.split("|")
        gold = gold_senses[0]
        alternatives = [sense for sense in attested[word] if sense not in gold_senses and sense != predicted]
        if not alternatives:
            continue
        pos = pos_of_word[word]
        to_predicted = sense_pair_relatedness(gold, predicted, pos)
        to_alternatives = [score for score in (sense_pair_relatedness(gold, a, pos) for a in alternatives)
                           if not np.isnan(score)]
        if np.isnan(to_predicted) or not to_alternatives:
            continue
        rows.append({
            "target_word": word,
            "pos": pos,
            "relatedness_measure": "path_similarity" if pos in ("NOUN", "VERB") else "extended_gloss_overlap",
            "predicted_is_mfs": predicted == mfs_of_word[word],
            "relatedness_to_predicted": to_predicted,
            "mean_relatedness_to_alternatives": float(np.mean(to_alternatives)),
            "predicted_is_most_related": to_predicted >= max(to_alternatives),
            # Chance of the predicted sense being the most related if errors ignored relatedness
            "chance_most_related": 1 / (len(to_alternatives) + 1),
        })
    return pd.DataFrame(rows)


def build_relatedness_table(instances_by_word, target_words):
    table = pd.DataFrame([describe_word_relatedness(word, instances_by_word[word]) for word in target_words])
    # Path similarity and gloss overlap live on different scales, so standardise within each measure
    for column in ("mean_pairwise_relatedness", "frequency_weighted_relatedness", "top_two_sense_relatedness"):
        grouped = table.groupby("relatedness_measure")[column]
        table[f"{column}_z"] = (table[column] - grouped.transform("mean")) / grouped.transform("std")
    return table
