"""Reader for corpora in the Raganato et al. (2017) unified XML format."""

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass
class Token:
    form: str
    lemma: str
    pos: str
    instance_id: str | None = None


@dataclass
class Sentence:
    sentence_id: str
    document_id: str
    tokens: list[Token]


@dataclass
class Instance:
    instance_id: str
    document_id: str
    sentence_id: str
    target_index: int
    lemma: str
    pos: str
    gold_senses: list[str]

    @property
    def target_word(self):
        return f"{self.lemma}.{self.pos}"

    @property
    def sense(self):
        # The first gold key is used as the training label
        return self.gold_senses[0]


@dataclass
class Corpus:
    name: str
    sentences: dict[str, Sentence] = field(default_factory=dict)
    instances: list[Instance] = field(default_factory=list)

    def sentence_of(self, instance):
        return self.sentences[instance.sentence_id]


def read_gold_keys(gold_key_path):
    gold_senses = {}
    with open(gold_key_path, encoding="utf-8") as gold_file:
        for line in gold_file:
            parts = line.split()
            if len(parts) >= 2:
                gold_senses[parts[0]] = parts[1:]
    return gold_senses


def read_corpus(name, xml_path, gold_key_path):
    gold_senses = read_gold_keys(gold_key_path)
    corpus = Corpus(name)

    for text in ElementTree.parse(Path(xml_path)).getroot().iter("text"):
        document_id = text.get("id")
        for sentence_element in text.iter("sentence"):
            sentence_id = sentence_element.get("id")
            tokens = []
            for position, element in enumerate(sentence_element):
                instance_id = element.get("id") if element.tag == "instance" else None
                token = Token(
                    form=element.text or "",
                    lemma=element.get("lemma", element.text or "").lower(),
                    pos=element.get("pos", "X"),
                    instance_id=instance_id,
                )
                tokens.append(token)
                if instance_id in gold_senses:
                    corpus.instances.append(Instance(
                        instance_id=instance_id,
                        document_id=document_id,
                        sentence_id=sentence_id,
                        target_index=position,
                        lemma=token.lemma,
                        pos=token.pos,
                        gold_senses=gold_senses[instance_id],
                    ))
            corpus.sentences[sentence_id] = Sentence(sentence_id, document_id, tokens)

    return corpus
