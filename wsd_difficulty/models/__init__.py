from .. import config
from ..wordnet_utils import split_target_word
from .decision_list import DecisionList
from .logistic_regression import LogisticRegression, tune_logistic_regression
from .most_frequent_sense import MostFrequentSense
from .naive_bayes import NaiveBayes, tune_naive_bayes
from .simplified_lesk import SimplifiedLesk

MODEL_NAMES = ("mfs", "lesk", "naive_bayes", "logistic_regression", "decision_list")

# These models ignore the feature sets, so they are run once per word
CONTEXT_FREE_MODELS = {"mfs", "lesk"}


def train_model(model_name, target_word, feature_lists, senses, documents):
    if model_name == "mfs":
        model = MostFrequentSense()
    elif model_name == "lesk":
        model = SimplifiedLesk(*split_target_word(target_word))
    elif model_name == "naive_bayes":
        smoothing, temperature = tune_naive_bayes(feature_lists, senses, documents,
                                                  config.SMOOTHING_GRID, config.TEMPERATURE_GRID)
        model = NaiveBayes(smoothing, temperature)
    elif model_name == "logistic_regression":
        l2_strength, epochs = tune_logistic_regression(feature_lists, senses, documents, config.L2_GRID,
                                                       config.EPOCH_CHECKPOINTS, seed=config.RANDOM_SEED)
        model = LogisticRegression(l2_strength=l2_strength, epochs=epochs, seed=config.RANDOM_SEED)
    elif model_name == "decision_list":
        model = DecisionList()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.fit(feature_lists, senses)
