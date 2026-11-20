"""Discourse context of a target instance: other mentions in the document, lexical chains, RST position."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from nltk.corpus import wordnet

from .rst import rst_description, rst_features, tree_distance_bucket
from .wordnet_utils import WORDNET_POS, candidate_sense_keys, synset_of_sense_key

CHAIN_POS = ("NOUN", "VERB")


@dataclass
class OtherMention:
    """Another occurrence of the same target word in the same document."""
    instance_id: str
    gold_sense: str
    sentence_gap: int            # other sentence number minus this one
    comes_before: bool
    tree_distance: int | None    # edges between the two EDUs, if RST trees exist


def sentence_gap_bucket(gap):
    gap = abs(gap)
    if gap == 0:
        return "same_sentence"
    if gap == 1:
        return "adjacent"
    if gap <= 5:
        return "near"
    if gap <= 20:
        return "middle"
    return "far"


def previous_mention(mentions):
    earlier = [mention for mention in mentions if mention.comes_before]
    return max(earlier, key=lambda mention: mention.sentence_gap) if earlier else None


def previous_mention_features(mentions, sense_of, use_rst):
    mention = previous_mention(mentions)
    if mention is None:
        return ["previous:none"]
    sense = sense_of(mention)
    if sense is None:
        return ["previous:unknown"]
    features = [f"previous:sense={sense}", f"previous:sense_{sentence_gap_bucket(mention.sentence_gap)}={sense}"]
    if use_rst and mention.tree_distance is not None:
        features.append(f"rst:previous_sense_{tree_distance_bucket(mention.tree_distance)}={sense}")
    return features


def document_sense_features(mentions, sense_of):
    senses = [sense for sense in map(sense_of, mentions) if sense is not None]
    if not senses:
        return ["document:no_other_mentions"]
    counts = Counter(senses)
    majority_sense, majority_count = counts.most_common(1)[0]
    # One feature per other mention, so Naive Bayes sees counts; LR treats them as present/absent
    features = [f"document:sense={sense}" for sense in senses]
    if majority_count > len(senses) / 2:
        features.append(f"document:majority={majority_sense}")
    return features


def dynamic_discourse_features(example, feature_types, sense_of):
    """Features that depend on the senses of other mentions (gold in training, predicted or gold in testing)."""
    features = []
    if "previous_mention" in feature_types:
        features += previous_mention_features(example.other_mentions, sense_of, use_rst="rst" in feature_types)
    if "document_senses" in feature_types:
        features += document_sense_features(example.other_mentions, sense_of)
    return features


# ---- Lexical chains ----

@lru_cache(maxsize=None)
def chain_neighbourhood(synset):
    """Synsets close enough to count as the same lexical chain (Galley & McKeown 2003 style)."""
    related = {synset}
    hypernyms = synset.hypernyms() + synset.instance_hypernyms()
    related.update(hypernyms)
    related.update(synset.hyponyms() + synset.instance_hyponyms())
    for hypernym in hypernyms:
        related.update(hypernym.hyponyms())  # siblings
    related.update(synset.entailments() + synset.similar_tos())
    for lemma in synset.lemmas():
        related.update(form.synset() for form in lemma.derivationally_related_forms())
    return frozenset(related)


@lru_cache(maxsize=None)
def synsets_of_lemma(lemma, pos):
    return tuple(wordnet.synsets(lemma, pos=WORDNET_POS[pos]))


class DocumentLexicon:
    """Which content lemmas occur in a document, and in which sentences."""

    def __init__(self, sentences):
        self.sentences_of_lemma = defaultdict(set)
        self.lemmas_of_synset = defaultdict(set)
        for sentence in sentences:
            for token in sentence.tokens:
                if token.pos in CHAIN_POS and token.lemma.replace("_", "").isalpha():
                    self.sentences_of_lemma[(token.lemma, token.pos)].add(sentence.sentence_id)
        for lemma, pos in self.sentences_of_lemma:
            for synset in synsets_of_lemma(lemma, pos):
                self.lemmas_of_synset[synset].add((lemma, pos))

    def chain_support(self, sense_key, target_lemma, sentence_id):
        """Distinct lemmas elsewhere in the document that are WordNet-close to this sense."""
        synset = synset_of_sense_key(sense_key)
        if synset is None:
            return 0
        supporters = set()
        for related in chain_neighbourhood(synset):
            supporters |= self.lemmas_of_synset.get(related, set())
        return sum(
            1 for lemma, pos in supporters
            if lemma != target_lemma and self.sentences_of_lemma[(lemma, pos)] - {sentence_id}
        )


def support_bucket(count):
    return "1" if count == 1 else "2-3" if count <= 3 else "4-7" if count <= 7 else "8+"


def lexical_chain_features(support_by_sense):
    supported = {sense: count for sense, count in support_by_sense.items() if count > 0}
    if not supported:
        return ["chain:none"]
    features = [f"chain:support_{support_bucket(count)}={sense}" for sense, count in supported.items()]
    best = max(supported.values())
    leaders = [sense for sense, count in supported.items() if count == best]
    if len(leaders) == 1:
        features.append(f"chain:strongest={leaders[0]}")
    return features


# ---- Putting it together for a corpus ----

def build_discourse_context(corpus, instances, rst_documents):
    """Return {instance_id: (static features, other mentions, descriptors)} for the given instances."""
    sentences_of_document = defaultdict(list)
    for sentence in corpus.sentences.values():
        sentences_of_document[sentence.document_id].append(sentence)
    sentence_number = {sentence.sentence_id: number
                       for sentences in sentences_of_document.values()
                       for number, sentence in enumerate(sentences)}

    lexicons = {}
    mentions_by_group = defaultdict(list)
    for instance in instances:
        mentions_by_group[(instance.document_id, instance.target_word)].append(instance)

    context = {}
    for instance in instances:
        document_id = instance.document_id
        if document_id not in lexicons:
            lexicons[document_id] = DocumentLexicon(sentences_of_document[document_id])
        sentence = corpus.sentences[instance.sentence_id]
        rst_document = rst_documents.get(document_id)

        static_features, descriptors = {}, {
            "sentence_length": len(sentence.tokens),
            "sentence_number": sentence_number[instance.sentence_id],
        }

        # Lexical chains over every WordNet sense of the target, plus any sense seen in the corpus
        candidates = set(candidate_sense_keys(instance.lemma, instance.pos)) | set(instance.gold_senses)
        support = {sense: lexicons[document_id].chain_support(sense, instance.lemma, instance.sentence_id)
                   for sense in candidates}
        static_features["lexical_chains"] = lexical_chain_features(support)
        descriptors["max_chain_support"] = max(support.values(), default=0)
        descriptors["gold_sense_chain_support"] = max(support.get(sense, 0) for sense in instance.gold_senses)
        descriptors["gold_sense_strongest_in_chain"] = (
            descriptors["max_chain_support"] > 0
            and descriptors["gold_sense_chain_support"] == descriptors["max_chain_support"])

        if rst_document:
            description = rst_description(rst_document, instance.sentence_id, instance.target_index)
            static_features["rst"] = rst_features(description)
            descriptors.update(description)
        else:
            static_features["rst"] = []

        own_order = (sentence_number[instance.sentence_id], instance.target_index)
        mentions = []
        for other in mentions_by_group[(document_id, instance.target_word)]:
            if other.instance_id == instance.instance_id:
                continue
            other_order = (sentence_number[other.sentence_id], other.target_index)
            tree_distance = None
            if rst_document:
                tree_distance = rst_document.tree.distance(
                    rst_document.edu_of_token[instance.sentence_id][instance.target_index],
                    rst_document.edu_of_token[other.sentence_id][other.target_index])
            mentions.append(OtherMention(other.instance_id, other.sense, other_order[0] - own_order[0],
                                         other_order < own_order, tree_distance))

        previous = previous_mention(mentions)
        descriptors["mentions_in_document"] = len(mentions) + 1
        descriptors["has_previous_mention"] = previous is not None
        descriptors["previous_mention_gap"] = abs(previous.sentence_gap) if previous else None
        descriptors["previous_mention_tree_distance"] = previous.tree_distance if previous else None
        descriptors["previous_mention_same_sense"] = (previous.gold_sense in instance.gold_senses) if previous else None
        context[instance.instance_id] = (static_features, mentions, descriptors)
    return context
