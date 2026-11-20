"""Per-word cross-validation on SemCor and external validation on the test sets."""

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from .discourse import dynamic_discourse_features
from .features import combine_feature_types
from .models import CONTEXT_FREE_MODELS, train_model

CROSS_VALIDATION = "semcor_cv"


@dataclass
class LabelledExample:
    instance_id: str
    document_id: str
    gold_senses: list[str]
    features: dict[str, list[str]]
    fold: int = -1
    evaluation: str = CROSS_VALIDATION
    other_mentions: list = field(default_factory=list)
    descriptors: dict = field(default_factory=dict)

    @property
    def sense(self):
        return self.gold_senses[0]


@dataclass
class WordJob:
    target_word: str
    training_examples: list[LabelledExample]
    external_examples: list[LabelledExample] = field(default_factory=list)
    model_names: tuple = ()
    feature_sets: dict = field(default_factory=dict)
    # First-stage held-out predicted senses of this word's instances, used for other mentions
    predicted_senses: dict = field(default_factory=dict)


def model_configurations(model_names, feature_sets):
    for model_name in model_names:
        if model_name in CONTEXT_FREE_MODELS:
            yield model_name, "none", ()
        else:
            for feature_set_name, feature_types in feature_sets.items():
                yield model_name, feature_set_name, feature_types


def gold_sense_of(mention):
    return mention.gold_sense


def features_for(example, model_name, feature_types, sense_of=gold_sense_of):
    if model_name == "lesk":
        return example.features["lesk_context"]
    static_types = [feature_type for feature_type in feature_types if feature_type in example.features]
    return combine_feature_types(example.features, static_types) + dynamic_discourse_features(
        example, feature_types, sense_of)


def train_and_predict(job, model_name, feature_set_name, feature_types, train_examples, test_examples, fold):
    # Realistic condition: other mentions carry the first stage's held-out predictions in training and test
    # documents alike, so the model never learns to trust a cleaner signal than it sees at test time.
    # Oracle condition: gold senses on both sides.
    if "oracle_senses" in feature_types:
        sense_of = gold_sense_of
    else:
        def sense_of(mention):
            return job.predicted_senses.get(mention.instance_id)

    model = train_model(
        model_name,
        job.target_word,
        [features_for(example, model_name, feature_types, sense_of) for example in train_examples],
        [example.sense for example in train_examples],
        [example.document_id for example in train_examples],
    )
    training_senses = {example.sense for example in train_examples}

    rows = []
    for example in test_examples:
        features = features_for(example, model_name, feature_types, sense_of)
        distribution = model.predict_distribution(features)
        predicted_sense = model.predict(features)
        rows.append({
            "evaluation": example.evaluation,
            "fold": fold,
            "target_word": job.target_word,
            "instance_id": example.instance_id,
            "document_id": example.document_id,
            "model": model_name,
            "feature_set": feature_set_name,
            "gold_senses": "|".join(example.gold_senses),
            "predicted_sense": predicted_sense,
            "correct": predicted_sense in example.gold_senses,
            "gold_seen_in_training": any(sense in training_senses for sense in example.gold_senses),
            "predicted_probability": distribution[predicted_sense] if distribution else math.nan,
            "gold_probability": max(distribution.get(sense, 0.0) for sense in example.gold_senses) if distribution else math.nan,
        })
    return rows


def evaluate_target_word(job):
    rows = []
    configurations = list(model_configurations(job.model_names, job.feature_sets))

    folds = sorted({example.fold for example in job.training_examples})
    for fold in folds:
        train_examples = [example for example in job.training_examples if example.fold != fold]
        test_examples = [example for example in job.training_examples if example.fold == fold]
        if not train_examples or not test_examples:
            continue
        for model_name, feature_set_name, feature_types in configurations:
            rows += train_and_predict(job, model_name, feature_set_name, feature_types, train_examples, test_examples, fold)

    # External validation: train on all of SemCor, test on the standard test sets
    if job.external_examples:
        for model_name, feature_set_name, feature_types in configurations:
            rows += train_and_predict(job, model_name, feature_set_name, feature_types,
                                      job.training_examples, job.external_examples, fold=-1)
    return rows


def checkpoint_path(checkpoint_dir, target_word):
    return checkpoint_dir / f"{target_word.replace('/', '_')}.csv.gz"


def read_checkpoint(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_experiments(jobs, workers=1, checkpoint_dir=None):
    """Evaluate every word; with a checkpoint directory each word is saved as it finishes and skipped on rerun."""
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        done = [job for job in jobs if checkpoint_path(checkpoint_dir, job.target_word).exists()]
        jobs = [job for job in jobs if not checkpoint_path(checkpoint_dir, job.target_word).exists()]
        if done:
            print(f"  {len(done)} target words already done, {len(jobs)} to go", flush=True)

    tables = []
    started = time.time()

    def collect(job, rows, finished):
        table = pd.DataFrame(rows)
        if checkpoint_dir is not None:
            table.to_csv(checkpoint_path(checkpoint_dir, job.target_word), index=False)
        else:
            tables.append(table)
        if finished % 50 == 0 or finished == len(jobs) or len(job.training_examples) >= 1000:
            minutes = (time.time() - started) / 60
            print(f"  evaluated {finished}/{len(jobs)} target words ({job.target_word}, "
                  f"{len(job.training_examples)} instances) after {minutes:.1f} min", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Collect in completion order so one slow word does not hold back the rest
            futures = {executor.submit(evaluate_target_word, job): job for job in jobs}
            for finished, future in enumerate(as_completed(futures), 1):
                collect(futures[future], future.result(), finished)
    else:
        for finished, job in enumerate(jobs, 1):
            collect(job, evaluate_target_word(job), finished)

    if checkpoint_dir is not None:
        # A word whose instances all sit in one document has no held-out fold and leaves an empty file
        tables = [read_checkpoint(path) for path in sorted(checkpoint_dir.glob("*.csv.gz"))]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
