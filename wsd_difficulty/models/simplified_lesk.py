import re
from collections import Counter
from functools import lru_cache

from nltk.corpus import wordnet
from spacy.lang.en.stop_words import STOP_WORDS

from ..wordnet_utils import candidate_sense_keys, synset_of_sense_key
from .base import SenseClassifier


def normalise_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {wordnet.morphy(word) or word for word in words if word not in STOP_WORDS}


@lru_cache(maxsize=None)
def sense_signature(sense_key):
    """Gloss words, example words and synonyms of a sense."""
    synset = synset_of_sense_key(sense_key)
    if synset is None:
        return frozenset()
    text = " ".join([synset.definition(), *synset.examples(), *(name.replace("_", " ") for name in synset.lemma_names())])
    return frozenset(normalise_words(text))


class SimplifiedLesk(SenseClassifier):
    """Picks the WordNet sense whose signature overlaps most with the sentence's content lemmas.

    Ties (including zero overlap) go to the sense seen most often in training, then WordNet order.
    """

    def __init__(self, lemma, pos):
        self.candidates = candidate_sense_keys(lemma, pos)

    def fit(self, feature_lists, senses):
        self.training_counts = Counter(senses)
        # Gold senses missing from WordNet's candidate list are still allowed
        self.candidates = tuple(dict.fromkeys([*self.candidates, *self.training_counts]))
        return self

    def overlap(self, sense_key, context_words):
        return len(sense_signature(sense_key) & context_words)

    def predict(self, context_lemmas):
        context_words = set(context_lemmas)
        wordnet_rank = {sense: rank for rank, sense in enumerate(self.candidates)}
        return max(
            self.candidates,
            key=lambda sense: (self.overlap(sense, context_words), self.training_counts[sense], -wordnet_rank[sense]),
        )
