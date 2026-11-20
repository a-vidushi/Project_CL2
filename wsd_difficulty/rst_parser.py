"""Run the DMRST parser (Liu et al., 2021) over whole documents.

Needs the optional `rst` dependency group (torch, transformers) and the checkpoint in data/models/.
"""

import json
import sys
import time

from . import config


def pick_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dmrst(device_name=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device_name) if device_name else pick_device(torch)
    # DMRST hard-codes .cuda(); route those calls to whichever device we have
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor.to(device)
    torch.nn.Module.cuda = lambda module, *args, **kwargs: module.to(device)

    sys.path.insert(0, str(config.DMRST_DIR))
    from model_depth import ParsingNet

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    language_model = AutoModel.from_pretrained("xlm-roberta-base").to(device)
    for parameter in language_model.parameters():
        parameter.requires_grad = False

    model = ParsingNet(language_model, bert_tokenizer=tokenizer).to(device)
    state = torch.load(config.DMRST_CHECKPOINT, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Newer transformers no longer store position_ids, everything else must match
    missing = [key for key in missing if not key.endswith("position_ids")]
    unexpected = [key for key in unexpected if not key.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    model.eval()
    print(f"  DMRST loaded on {device}")
    return torch, model, tokenizer


def subword_tokens(tokenizer, sentences):
    """Tokenise word by word so each corpus token knows its last subword position."""
    subwords, last_subword_of_token = [], {}
    for sentence in sentences:
        positions = []
        for token in sentence.tokens:
            pieces = tokenizer.tokenize(token.form.replace("_", " ")) if token.form.strip() else []
            subwords.extend(pieces)
            positions.append(len(subwords) - 1 if subwords else 0)
        last_subword_of_token[sentence.sentence_id] = positions
    return subwords, last_subword_of_token


def parse_document(torch, model, tokenizer, sentences):
    subwords, last_subword_of_token = subword_tokens(tokenizer, sentences)
    with torch.no_grad():
        _, _, span_batch, _, edu_breaks_batch = model.TestingLoss(
            [subwords], input_EDU_breaks=None, LabelIndex=None, ParsingIndex=None,
            GenerateTree=True, use_pred_segmentation=True)
    edu_breaks = edu_breaks_batch[0]

    def edu_of_subword(position):
        for edu, last_position in enumerate(edu_breaks):
            if position <= last_position:
                return edu
        return len(edu_breaks) - 1

    return {
        "number_of_edus": len(edu_breaks),
        "spans": span_batch[0][0],
        "edu_of_token": {sentence_id: [edu_of_subword(p) for p in positions]
                         for sentence_id, positions in last_subword_of_token.items()},
    }


def sentences_by_document(corpus):
    documents = {}
    for sentence in corpus.sentences.values():
        documents.setdefault(sentence.document_id, []).append(sentence)
    return documents


def parse_corpus(corpus, output_path, device_name=None, limit=None):
    """Parse every document, saving after each one so an interrupted run can resume."""
    parsed = json.loads(output_path.read_text()) if output_path.exists() else {}
    documents = sentences_by_document(corpus)
    remaining = [document for document in documents if document not in parsed][:limit]
    print(f"  {len(parsed)} documents already parsed, {len(remaining)} to go")
    if not remaining:
        return parsed

    torch, model, tokenizer = load_dmrst(device_name)
    for count, document_id in enumerate(remaining, 1):
        started = time.time()
        parsed[document_id] = parse_document(torch, model, tokenizer, documents[document_id])
        output_path.write_text(json.dumps(parsed))
        print(f"  [{count}/{len(remaining)}] {document_id}: {parsed[document_id]['number_of_edus']} EDUs "
              f"in {time.time() - started:.1f}s", flush=True)
    return parsed
