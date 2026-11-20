from functools import lru_cache

from nltk.corpus import wordnet

WORDNET_POS = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}


def split_target_word(target_word):
    lemma, pos = target_word.rsplit(".", 1)
    return lemma, pos


@lru_cache(maxsize=None)
def candidate_sense_keys(lemma, pos):
    """WordNet sense keys of a lemma in WordNet order (most frequent first)."""
    keys = []
    for synset in wordnet.synsets(lemma, pos=WORDNET_POS.get(pos)):
        for wordnet_lemma in synset.lemmas():
            if wordnet_lemma.name().lower() == lemma.lower():
                keys.append(wordnet_lemma.key())
                break
    return tuple(keys)


@lru_cache(maxsize=None)
def synset_of_sense_key(sense_key):
    try:
        return wordnet.lemma_from_key(sense_key).synset()
    except Exception:
        # Adjective satellites are sometimes keyed as head adjectives (%3 vs %5)
        for alternative in (sense_key.replace("%3", "%5"), sense_key.replace("%5", "%3")):
            try:
                return wordnet.lemma_from_key(alternative).synset()
            except Exception:
                continue
    return None


def wordnet_polysemy(lemma, pos):
    return len(candidate_sense_keys(lemma, pos))
