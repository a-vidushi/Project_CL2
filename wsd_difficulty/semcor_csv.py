"""Reader for raw SemCor 3.0 exported as CSV (one row per token, Hugging Face layout)."""

from pathlib import Path

import pandas as pd

from .corpus import Corpus, Instance, Sentence, Token
from .wordnet_utils import synset_of_sense_key

SPLITS = ("brown1", "brown2", "brownv")

# The first digit of a lexsn is the WordNet synset type
POS_OF_SYNSET_TYPE = {"1": "NOUN", "2": "VERB", "3": "ADJ", "4": "ADV", "5": "ADJ"}


def coarse_pos(penn_tag, row_type):
    if row_type == "punc":
        return "."
    for prefix, pos in (("NN", "NOUN"), ("VB", "VERB"), ("JJ", "ADJ"), ("RB", "ADV")):
        if penn_tag.startswith(prefix):
            return pos
    return "X"


def missing(value):
    return value in ("None", "", "nan")


def gold_sense_keys(lemma, lexsn_field):
    """Sense keys for a token, keeping only those that exist in WordNet 3.0."""
    keys = [f"{lemma.lower()}%{lexsn}" for lexsn in lexsn_field.split(";")]
    return [key for key in keys if synset_of_sense_key(key) is not None]


def read_semcor_csv(csv_dir, name="semcor"):
    corpus = Corpus(name)
    for split in SPLITS:
        path = Path(csv_dir) / f"{split}-00000-of-00001.csv"
        if not path.exists():
            print(f"  SemCor split {split} not found at {path}, skipping")
            continue
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)

        for (tagfile, paragraph, sentence_number), sentence_rows in rows.groupby(
                ["tagfile", "pnum", "snum"], sort=False):
            document_id = f"{split}/{tagfile}"
            sentence_id = f"{document_id}.p{paragraph}.s{sentence_number}"
            tokens = []
            for position, row in enumerate(sentence_rows.itertuples(index=False)):
                form = row.value
                is_tagged = row.tag == "wf" and not missing(row.lemma) and not missing(row.lexsn)
                senses = gold_sense_keys(row.lemma, row.lexsn) if is_tagged else []
                instance_id = f"{sentence_id}.t{position}" if senses else None

                lemma = row.lemma.lower() if not missing(row.lemma) else form.lower()
                pos = POS_OF_SYNSET_TYPE[row.lexsn[0]] if senses else coarse_pos(row.pos, row.tag)
                tokens.append(Token(form=form, lemma=lemma, pos=pos, instance_id=instance_id))

                if senses:
                    corpus.instances.append(Instance(
                        instance_id=instance_id,
                        document_id=document_id,
                        sentence_id=sentence_id,
                        target_index=position,
                        lemma=lemma,
                        pos=pos,
                        gold_senses=senses,
                    ))
            corpus.sentences[sentence_id] = Sentence(sentence_id, document_id, tokens)
    return corpus
