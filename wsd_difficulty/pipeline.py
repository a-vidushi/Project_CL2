"""Pipeline stages: parse RST, prepare data, run sentence-level and discourse experiments, analyse."""

import pickle
import random

import pandas as pd

from . import analysis, classification_metrics, config, correctness_regression, discourse_analysis
from .corpus import read_corpus
from .discourse import build_discourse_context
from .experiments import CROSS_VALIDATION, LabelledExample, WordJob, run_experiments
from .features import build_feature_table
from .folds import assign_document_folds
from .information import build_feature_mutual_information_table, summarise_top_features
from .lemma_selection import group_by_target_word, select_target_words
from .parsing import load_parser, parse_sentences
from .relatedness import build_relatedness_table, confusion_relatedness
from .rst import load_rst_documents
from .semcor_csv import read_semcor_csv
from .sense_statistics import build_word_table, summarise_by_group

PREPARED_DATA_FILE = "prepared.pkl"
PREDICTIONS_FILE = "predictions.csv.gz"
DISCOURSE_PREDICTIONS_FILE = "predictions_discourse.csv.gz"
DISCOURSE_CHECKPOINT_DIR = "discourse_by_word"


def load_corpora(framework_dir):
    semcor_xml, semcor_gold_keys = config.training_corpus_paths(framework_dir)
    if semcor_xml.exists():
        semcor = read_corpus(config.TRAINING_CORPUS_NAME, semcor_xml, semcor_gold_keys)
    else:
        print(f"  {semcor_xml} not found, reading SemCor CSVs from {config.SEMCOR_CSV_DIR}")
        semcor = read_semcor_csv(config.SEMCOR_CSV_DIR, config.TRAINING_CORPUS_NAME)
    print(f"  SemCor: {len(semcor.sentences)} sentences, {len(semcor.instances)} sense-tagged instances")
    test_corpora = []
    for name in config.TEST_SET_NAMES:
        xml_path, gold_key_path = config.test_set_paths(name, framework_dir)
        if xml_path.exists() and gold_key_path.exists():
            test_corpora.append(read_corpus(name, xml_path, gold_key_path))
        else:
            print(f"  test set {name} not found, skipping")
    return semcor, test_corpora


def parse_rst(framework_dir=config.FRAMEWORK_DIR, device=None, limit=None):
    from .rst_parser import parse_corpus

    print("Parsing SemCor documents with DMRST")
    semcor, _ = load_corpora(framework_dir)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parse_corpus(semcor, config.PROCESSED_DIR / config.RST_TREES_FILE, device, limit)


def labelled_examples(corpus, target_words, parser, rst_documents, fold_of_document=None):
    """Parse the sentences that contain target words and turn their instances into feature-bearing examples."""
    instances = [instance for instance in corpus.instances if instance.target_word in target_words]
    sentence_ids = {instance.sentence_id for instance in instances}
    parses = parse_sentences([corpus.sentences[sentence_id] for sentence_id in sorted(sentence_ids)], parser)
    feature_table = build_feature_table(instances, corpus.sentences, parses)
    discourse_context = build_discourse_context(corpus, instances, rst_documents)

    examples_by_word = {}
    for instance in instances:
        static_features, other_mentions, descriptors = discourse_context[instance.instance_id]
        examples_by_word.setdefault(instance.target_word, []).append(LabelledExample(
            instance_id=instance.instance_id,
            document_id=instance.document_id,
            gold_senses=instance.gold_senses,
            features={**feature_table[instance.instance_id], **static_features},
            fold=fold_of_document[instance.document_id] if fold_of_document else -1,
            evaluation=CROSS_VALIDATION if fold_of_document else corpus.name,
            other_mentions=other_mentions,
            descriptors=descriptors,
        ))
    return examples_by_word


def corpus_overview(semcor, test_corpora, target_words):
    rows = []
    for corpus in [semcor, *test_corpora]:
        covered = [instance for instance in corpus.instances if instance.target_word in target_words]
        rows.append({"corpus": corpus.name, "documents": len({i.document_id for i in corpus.instances}),
                     "instances": len(corpus.instances), "instances_of_target_words": len(covered),
                     "target_words_present": len({i.target_word for i in covered})})
    return pd.DataFrame(rows)


def instance_table(examples_by_word):
    return pd.DataFrame([
        {"instance_id": example.instance_id, "target_word": word, "document_id": example.document_id,
         **example.descriptors}
        for word, examples in examples_by_word.items() for example in examples
    ])


def prepare(framework_dir=config.FRAMEWORK_DIR, max_target_words=None):
    print("Reading corpora")
    semcor, test_corpora = load_corpora(framework_dir)

    instances_by_word = group_by_target_word(semcor.instances)
    target_words = select_target_words(semcor.instances)
    print(f"  {len(target_words)} target words pass the selection thresholds")
    if max_target_words and max_target_words < len(target_words):
        # A random sample keeps the POS and entropy mix of the full set
        target_words = sorted(random.Random(config.RANDOM_SEED).sample(target_words, max_target_words))
        print(f"  using a sample of {len(target_words)}")
    selected = set(target_words)

    print("Computing sense statistics and relatedness")
    word_table = build_word_table(instances_by_word, target_words)
    relatedness_table = build_relatedness_table(instances_by_word, target_words)
    word_table = word_table.merge(relatedness_table, on="target_word")

    rst_documents = load_rst_documents(config.PROCESSED_DIR / config.RST_TREES_FILE)
    semcor_documents = {instance.document_id for instance in semcor.instances}
    has_rst_trees = bool(rst_documents) and semcor_documents <= set(rst_documents)
    if rst_documents and not has_rst_trees:
        print(f"  RST trees cover only {len(semcor_documents & set(rst_documents))}/{len(semcor_documents)} "
              "documents; RST features are left out until parse-rst finishes")
        rst_documents = {}
    print(f"  RST trees: {'yes' if has_rst_trees else 'no'}")

    print("Parsing, extracting sentence and discourse features")
    parser = load_parser()
    fold_of_document = assign_document_folds(semcor.instances, config.NUMBER_OF_FOLDS, config.RANDOM_SEED)
    training_examples = labelled_examples(semcor, selected, parser, rst_documents, fold_of_document)
    external_examples = {}
    for test_corpus in test_corpora:
        for word, examples in labelled_examples(test_corpus, selected, parser, {}).items():
            external_examples.setdefault(word, []).extend(examples)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / PREPARED_DATA_FILE, "wb") as prepared_file:
        pickle.dump({"word_table": word_table, "training_examples": training_examples,
                     "external_examples": external_examples, "has_rst_trees": has_rst_trees}, prepared_file)

    results = config.RESULTS_DIR
    results.mkdir(parents=True, exist_ok=True)
    word_table.to_csv(results / "word_statistics.csv", index=False)
    instance_table(training_examples).to_csv(config.PROCESSED_DIR / "instance_descriptors.csv.gz", index=False)
    corpus_overview(semcor, test_corpora, selected).to_csv(results / "corpus_overview.csv", index=False)
    summarise_by_group(word_table, ["pos"]).to_csv(results / "entropy_by_pos.csv", index=False)
    summarise_by_group(word_table, ["entropy_tercile"]).to_csv(results / "entropy_by_tercile.csv", index=False)
    summarise_by_group(word_table, ["entropy_tercile", "pos"]).to_csv(
        results / "entropy_by_tercile_and_pos.csv", index=False)
    print(f"Prepared data saved to {config.PROCESSED_DIR}")


def load_prepared():
    with open(config.PROCESSED_DIR / PREPARED_DATA_FILE, "rb") as prepared_file:
        return pickle.load(prepared_file)


def build_jobs(prepared, model_names, feature_sets, predicted_senses=None):
    jobs = []
    for word, examples in prepared["training_examples"].items():
        external = prepared["external_examples"].get(word, [])
        word_predictions = {}
        if predicted_senses:
            word_predictions = {e.instance_id: predicted_senses[e.instance_id]
                                for e in examples + external if e.instance_id in predicted_senses}
        jobs.append(WordJob(word, examples, external, tuple(model_names), feature_sets, word_predictions))
    # Largest words first so parallel workers finish at about the same time
    jobs.sort(key=lambda job: len(job.training_examples), reverse=True)
    return jobs


def subsample_within_folds(examples, max_instances, seed=config.RANDOM_SEED):
    """Random sample of a word's instances, the same share from every fold."""
    if len(examples) <= max_instances:
        return examples
    share = max_instances / len(examples)
    generator = random.Random(seed)
    sample = []
    for fold in sorted({example.fold for example in examples}):
        in_fold = [example for example in examples if example.fold == fold]
        sample += generator.sample(in_fold, round(len(in_fold) * share))
    return sample


def run(model_names, feature_set_names, workers=1):
    prepared = load_prepared()
    feature_sets = {name: config.FEATURE_SETS[name] for name in feature_set_names}
    jobs = build_jobs(prepared, model_names, feature_sets)
    print(f"Running {len(model_names)} models x {len(feature_sets)} feature sets on {len(jobs)} target words")
    predictions = run_experiments(jobs, workers)
    predictions.to_csv(config.PROCESSED_DIR / PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to {config.PROCESSED_DIR / PREDICTIONS_FILE}")


def run_discourse(workers=1):
    """Second stage: other mentions carry the first-stage model's predicted senses at test time."""
    prepared = load_prepared()
    predictions = pd.read_csv(config.PROCESSED_DIR / PREDICTIONS_FILE)
    model_name, feature_set = config.FIRST_STAGE_MODEL
    first_stage = predictions[(predictions["model"] == model_name) & (predictions["feature_set"] == feature_set)]
    predicted_senses = dict(zip(first_stage["instance_id"], first_stage["predicted_sense"]))

    feature_sets = dict(config.DISCOURSE_FEATURE_SETS)
    if not prepared["has_rst_trees"]:
        print("  no RST trees: skipping RST feature sets")
        feature_sets = {name: types for name, types in feature_sets.items() if "rst" not in types}
    jobs = build_jobs(prepared, config.DISCOURSE_MODELS, feature_sets, predicted_senses)
    for job in jobs:
        if len(job.training_examples) > config.MAX_DISCOURSE_INSTANCES:
            print(f"  subsampling {job.target_word} from {len(job.training_examples)} "
                  f"to {config.MAX_DISCOURSE_INSTANCES} instances")
            # Other mentions keep their full set: only which instances are classified changes
            job.training_examples = subsample_within_folds(job.training_examples, config.MAX_DISCOURSE_INSTANCES)
    print(f"Running {len(config.DISCOURSE_MODELS)} models x {len(feature_sets)} discourse feature sets "
          f"on {len(jobs)} target words")
    discourse_predictions = run_experiments(jobs, workers, config.PROCESSED_DIR / DISCOURSE_CHECKPOINT_DIR)
    discourse_predictions.to_csv(config.PROCESSED_DIR / DISCOURSE_PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to {config.PROCESSED_DIR / DISCOURSE_PREDICTIONS_FILE}")


def load_all_predictions():
    tables = [pd.read_csv(config.PROCESSED_DIR / PREDICTIONS_FILE)]
    discourse_file = config.PROCESSED_DIR / DISCOURSE_PREDICTIONS_FILE
    if discourse_file.exists():
        tables.append(pd.read_csv(discourse_file))
    return pd.concat(tables, ignore_index=True)


def analyse():
    prepared = load_prepared()
    word_table = prepared["word_table"]
    predictions = load_all_predictions()
    instances = instance_table(prepared["training_examples"])
    results = config.RESULTS_DIR
    results.mkdir(parents=True, exist_ok=True)

    def save(table, name):
        table.to_csv(results / name, index=False)
        print(f"  wrote {name}")

    word_accuracy = analysis.word_accuracy_table(predictions)
    save(analysis.overall_accuracy_table(predictions), "accuracy_overall.csv")
    save(word_accuracy, "accuracy_by_word.csv")
    save(analysis.context_lift_by_stratum(word_accuracy, word_table), "context_lift_by_tercile_and_pos.csv")
    classification_metrics.write_metrics(predictions, word_table, results)

    # RQ1
    save(analysis.difficulty_correlations(word_accuracy, word_table), "rq1_correlations.csv")
    save(analysis.entropy_beyond_baseline_regressions(word_accuracy, word_table), "rq1_entropy_beyond_baseline.csv")
    save(analysis.collinearity_table(word_table), "rq1_variance_inflation_factors.csv")

    # RQ2
    gain_by_stratum, gain_versus_lift = analysis.information_gain_by_stratum(predictions, word_table, word_accuracy)
    save(gain_by_stratum, "rq2_information_gain_by_tercile_and_pos.csv")
    save(gain_versus_lift, "rq2_information_gain_vs_lift.csv")
    mutual_information = build_feature_mutual_information_table(prepared["training_examples"], config.FEATURE_TYPES)
    if not mutual_information.empty:
        save(mutual_information[mutual_information["rank_within_word"] <= 20], "rq2_top_features_by_mutual_information.csv")
        save(summarise_top_features(mutual_information), "rq2_mutual_information_by_context_type.csv")
    save(analysis.naive_bayes_versus_logistic_regression(predictions), "naive_bayes_vs_logistic_regression.csv")

    # RQ3
    confusions = confusion_relatedness(predictions, word_table, prepared["training_examples"])
    save(confusions, "rq3_error_confusions.csv")
    save(analysis.confusion_relatedness_summary(confusions), "rq3_confused_sense_relatedness.csv")

    # RQ4 and discourse
    consistency_by_word = discourse_analysis.document_consistency_by_word(prepared["training_examples"])
    save(consistency_by_word, "discourse_consistency_by_word.csv")
    save(discourse_analysis.summarise_consistency(consistency_by_word, word_table), "discourse_one_sense_per_discourse.csv")
    save(discourse_analysis.previous_mention_agreement(instances, word_table), "discourse_previous_mention_agreement.csv")
    if predictions["feature_set"].str.startswith("all+").any():
        save(discourse_analysis.discourse_gain_by_stratum(predictions, word_table), "rq4_discourse_gain_by_tercile_and_pos.csv")
        save(discourse_analysis.discourse_significance(predictions), "rq4_discourse_vs_sentence_mcnemar.csv")
        save(discourse_analysis.discourse_information_gain(predictions, word_table), "rq4_discourse_information_gain.csv")
        save(discourse_analysis.gain_by_mention_status(predictions, instances), "rq4_gain_by_previous_mention.csv")

    # Instance-level correctness regression (lemma-clustered standard errors)
    coefficients = correctness_regression.fit_correctness_models(predictions, instances, word_table)
    save(coefficients, "correctness_regression_coefficients.csv")

    save(analysis.external_accuracy_by_tercile(predictions, word_table), "external_accuracy_by_tercile.csv")
