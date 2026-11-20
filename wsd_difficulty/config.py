from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRAMEWORK_DIR = DATA_DIR / "raw" / "WSD_Evaluation_Framework"
# Fallback when the Raganato bundle is absent: raw SemCor 3.0 as CSV
SEMCOR_CSV_DIR = DATA_DIR / "raw" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAINING_CORPUS_NAME = "semcor"
TEST_SET_NAMES = ("senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015")

# Lemma selection thresholds from the proposal
MINIMUM_INSTANCES = 30
MINIMUM_FREQUENT_SENSES = 2
MINIMUM_INSTANCES_PER_SENSE = 5

NUMBER_OF_FOLDS = 10
RANDOM_SEED = 13

WINDOW_SIZE = 5
POS_WINDOW_SIZE = 3

FEATURE_TYPES = ("window", "collocations", "pos", "dependencies")

# Single context types (for RQ2) and the cumulative ladder from the proposal
FEATURE_SETS = {
    "window": ("window",),
    "collocations": ("collocations",),
    "pos": ("pos",),
    "dependencies": ("dependencies",),
    "window+collocations": ("window", "collocations"),
    "window+collocations+pos": ("window", "collocations", "pos"),
    "all": ("window", "collocations", "pos", "dependencies"),
}

SENTENCE_FEATURES = FEATURE_SETS["all"]

# Discourse feature sets, each added on top of all sentence-level features (RQ4).
# "oracle_senses" means other mentions carry their gold senses at test time instead of predicted ones.
DISCOURSE_FEATURE_TYPES = ("previous_mention", "document_senses", "lexical_chains", "rst")
DISCOURSE_FEATURE_SETS = {
    "all+previous_mention": SENTENCE_FEATURES + ("previous_mention",),
    "all+document_senses": SENTENCE_FEATURES + ("document_senses",),
    "all+lexical_chains": SENTENCE_FEATURES + ("lexical_chains",),
    "all+rst": SENTENCE_FEATURES + ("rst",),
    "all+cohesion": SENTENCE_FEATURES + ("previous_mention", "document_senses", "lexical_chains"),
    "all+discourse": SENTENCE_FEATURES + DISCOURSE_FEATURE_TYPES,
    "all+discourse_oracle": SENTENCE_FEATURES + DISCOURSE_FEATURE_TYPES + ("oracle_senses",),
}
DISCOURSE_MODELS = ("naive_bayes", "logistic_regression", "decision_list")

# Very frequent words (in practice only be.VERB) are subsampled to this many instances in the discourse
# stage, because their many same-document mentions make discourse features very slow to train on
MAX_DISCOURSE_INSTANCES = 3000

# The sentence-level model whose predictions stand in for the senses of other mentions
FIRST_STAGE_MODEL = ("logistic_regression", "all")

SMOOTHING_GRID = (0.01, 0.05, 0.1, 0.5, 1.0)
TEMPERATURE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
L2_GRID = (1e-5, 1e-4, 1e-3)
EPOCH_CHECKPOINTS = (5, 10, 20)

RST_TREES_FILE = "rst_trees.json"
DMRST_DIR = PROJECT_ROOT / "third_party" / "DMRST_Parser"
DMRST_CHECKPOINT = DATA_DIR / "models" / "multi_all_checkpoint.torchsave"


def training_corpus_paths(framework_dir=FRAMEWORK_DIR):
    folder = Path(framework_dir) / "Training_Corpora" / "SemCor"
    return folder / "semcor.data.xml", folder / "semcor.gold.key.txt"


def test_set_paths(name, framework_dir=FRAMEWORK_DIR):
    folder = Path(framework_dir) / "Evaluation_Datasets" / name
    return folder / f"{name}.data.xml", folder / f"{name}.gold.key.txt"
