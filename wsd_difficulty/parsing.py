"""spaCy POS tags and dependencies on the corpus's own (gold) tokenisation."""

from dataclasses import dataclass

import spacy
from spacy.tokens import Doc


@dataclass
class ParsedToken:
    tag: str
    dependency: str
    head_index: int


def load_parser(model_name="en_core_web_sm"):
    return spacy.load(model_name, disable=["ner", "lemmatizer"])


def parse_sentences(sentences, parser=None, batch_size=256):
    """Return {sentence_id: [ParsedToken, ...]} aligned one-to-one with the gold tokens."""
    parser = parser or load_parser()
    sentence_list = list(sentences)

    def gold_tokenised_docs():
        for sentence in sentence_list:
            # spaCy tokens cannot hold spaces, so multiword forms are joined with underscores
            words = [token.form.replace(" ", "_") or "_" for token in sentence.tokens]
            yield Doc(parser.vocab, words=words)

    parses = {}
    for sentence, doc in zip(sentence_list, parser.pipe(gold_tokenised_docs(), batch_size=batch_size)):
        parses[sentence.sentence_id] = [ParsedToken(token.tag_, token.dep_, token.head.i) for token in doc]
    return parses
