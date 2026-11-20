"""Context features for each target instance, grouped by context type."""

from spacy.lang.en.stop_words import STOP_WORDS

from . import config

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def is_content_lemma(token):
    return token.lemma.replace("_", "").isalpha() and token.lemma not in STOP_WORDS


def window_features(sentence, target_index, window_size=config.WINDOW_SIZE):
    start = max(0, target_index - window_size)
    end = min(len(sentence.tokens), target_index + window_size + 1)
    return sorted({
        f"window:{token.lemma}"
        for position, token in enumerate(sentence.tokens[start:end], start)
        if position != target_index and is_content_lemma(token)
    })


def collocation_features(sentence, target_index):
    """Yarowsky-style collocations over lowercased surface words at fixed offsets."""
    def word_at(offset):
        position = target_index + offset
        if 0 <= position < len(sentence.tokens):
            return sentence.tokens[position].form.lower()
        return "<boundary>"

    offset_patterns = [(-1,), (1,), (-2,), (2,), (-2, -1), (-1, 1), (1, 2)]
    return [
        f"collocation:{','.join(map(str, offsets))}={'_'.join(word_at(offset) for offset in offsets)}"
        for offsets in offset_patterns
    ]


def pos_features(sentence_parse, target_index, window_size=config.POS_WINDOW_SIZE):
    features = []
    for offset in range(-window_size, window_size + 1):
        position = target_index + offset
        if 0 <= position < len(sentence_parse):
            tag = sentence_parse[position].tag
        else:
            tag = "<boundary>"
        features.append(f"pos:{offset}={tag}")
    return features


def dependency_features(sentence, sentence_parse, target_index):
    target = sentence_parse[target_index]
    features = [f"dependency:relation={target.dependency}"]

    if target.head_index != target_index:
        head_lemma = sentence.tokens[target.head_index].lemma
        features.append(f"dependency:head={target.dependency}^{head_lemma}")

    for position, parsed_token in enumerate(sentence_parse):
        if parsed_token.head_index == target_index and position != target_index:
            child_lemma = sentence.tokens[position].lemma
            features.append(f"dependency:child={parsed_token.dependency}:{child_lemma}")
    return features


def lesk_context(sentence, target_index):
    """Every content lemma in the sentence, for simplified Lesk."""
    return sorted({
        token.lemma for position, token in enumerate(sentence.tokens)
        if position != target_index and is_content_lemma(token)
    })


def extract_instance_features(instance, sentence, sentence_parse):
    index = instance.target_index
    return {
        "window": window_features(sentence, index),
        "collocations": collocation_features(sentence, index),
        "pos": pos_features(sentence_parse, index),
        "dependencies": dependency_features(sentence, sentence_parse, index),
        "lesk_context": lesk_context(sentence, index),
    }


def build_feature_table(instances, sentences, parses):
    """Return {instance_id: {feature_type: [features]}}."""
    return {
        instance.instance_id: extract_instance_features(
            instance, sentences[instance.sentence_id], parses[instance.sentence_id]
        )
        for instance in instances
    }


def combine_feature_types(instance_features, feature_types):
    combined = []
    for feature_type in feature_types:
        combined.extend(instance_features[feature_type])
    return combined
