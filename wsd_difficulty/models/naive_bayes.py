import math
from collections import Counter, defaultdict

from ..folds import split_off_development_documents
from .base import SenseClassifier


class NaiveBayes(SenseClassifier):
    """Multinomial Naive Bayes over feature occurrences with add-k smoothing.

    Naive Bayes multiplies many correlated feature likelihoods, so its raw posteriors are far
    too confident. Dividing the log scores by a temperature > 1 (tuned on held-out documents)
    calibrates the distribution without changing which sense is predicted.
    """

    def __init__(self, smoothing=1.0, temperature=1.0):
        self.smoothing = smoothing
        self.temperature = temperature

    def fit(self, feature_lists, senses):
        self.sense_counts = Counter(senses)
        self.feature_counts = defaultdict(Counter)
        for features, sense in zip(feature_lists, senses):
            self.feature_counts[sense].update(features)

        self.vocabulary = {feature for counts in self.feature_counts.values() for feature in counts}
        self.total_feature_count = {sense: sum(counts.values()) for sense, counts in self.feature_counts.items()}
        return self

    def log_scores(self, features):
        k = self.smoothing
        vocabulary_size = len(self.vocabulary)
        number_of_senses = len(self.sense_counts)
        total_instances = sum(self.sense_counts.values())
        known_features = [feature for feature in features if feature in self.vocabulary]

        scores = {}
        for sense, sense_count in self.sense_counts.items():
            score = math.log((sense_count + k) / (total_instances + k * number_of_senses))
            denominator = self.total_feature_count.get(sense, 0) + k * vocabulary_size
            counts = self.feature_counts[sense]
            for feature in known_features:
                score += math.log((counts[feature] + k) / denominator)
            scores[sense] = score
        return scores

    def predict_distribution(self, features):
        return softmax_of_scores(self.log_scores(features), self.temperature)

    def predict(self, features):
        scores = self.log_scores(features)
        return max(scores, key=scores.get)


def softmax_of_scores(scores, temperature=1.0):
    best = max(scores.values())
    exponentiated = {sense: math.exp((score - best) / temperature) for sense, score in scores.items()}
    total = sum(exponentiated.values())
    return {sense: value / total for sense, value in exponentiated.items()}


def tune_naive_bayes(feature_lists, senses, documents, smoothing_grid, temperature_grid, default=(1.0, 1.0)):
    """Choose k by accuracy, then the temperature by log-loss, both on held-out training documents."""
    development_documents = split_off_development_documents(documents)
    train_rows = [i for i, document in enumerate(documents) if document not in development_documents]
    development_rows = [i for i, document in enumerate(documents) if document in development_documents]
    if not train_rows or not development_rows:
        return default

    train_features = [feature_lists[i] for i in train_rows]
    train_senses = [senses[i] for i in train_rows]

    best_model, best_accuracy = None, -1.0
    for smoothing in smoothing_grid:
        model = NaiveBayes(smoothing).fit(train_features, train_senses)
        accuracy = sum(model.predict(feature_lists[i]) == senses[i] for i in development_rows) / len(development_rows)
        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy

    # Only development instances whose sense was seen in training can be scored by log-loss
    scorable_rows = [i for i in development_rows if senses[i] in best_model.sense_counts]
    if not scorable_rows:
        return best_model.smoothing, default[1]
    scores_by_row = {i: best_model.log_scores(feature_lists[i]) for i in scorable_rows}

    def log_loss(temperature):
        return -sum(math.log(max(softmax_of_scores(scores_by_row[i], temperature)[senses[i]], 1e-12))
                    for i in scorable_rows)

    best_temperature = min(temperature_grid, key=log_loss)
    return best_model.smoothing, best_temperature
