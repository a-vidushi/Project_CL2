from collections import Counter

from .base import SenseClassifier


class MostFrequentSense(SenseClassifier):
    """Always predicts the training MFS; its distribution is the add-k smoothed sense prior."""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def fit(self, feature_lists, senses):
        counts = Counter(senses)
        total = len(senses) + self.smoothing * len(counts)
        self.prior = {sense: (count + self.smoothing) / total for sense, count in counts.items()}
        self.most_frequent_sense = counts.most_common(1)[0][0]
        return self

    def predict_distribution(self, features):
        return self.prior

    def predict(self, features):
        return self.most_frequent_sense
