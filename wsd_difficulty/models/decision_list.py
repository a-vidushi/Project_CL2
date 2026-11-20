import math
from collections import Counter, defaultdict

from .base import SenseClassifier


class DecisionList(SenseClassifier):
    """Yarowsky-style decision list: each feature votes for one sense, rules ranked by log-likelihood ratio.

    The highest-ranked rule whose feature is present decides; otherwise fall back to the MFS.
    """

    def __init__(self, smoothing=0.1, minimum_feature_count=2):
        self.smoothing = smoothing
        self.minimum_feature_count = minimum_feature_count

    def fit(self, feature_lists, senses):
        sense_counts = Counter(senses)
        self.most_frequent_sense = sense_counts.most_common(1)[0][0]

        counts_by_feature = defaultdict(Counter)
        for features, sense in zip(feature_lists, senses):
            for feature in set(features):
                counts_by_feature[feature][sense] += 1

        # rule for each feature: (log-likelihood ratio, sense)
        self.rules = {}
        for feature, counts in counts_by_feature.items():
            feature_total = sum(counts.values())
            if feature_total < self.minimum_feature_count:
                continue
            sense, count_with_sense = counts.most_common(1)[0]
            count_without_sense = feature_total - count_with_sense
            log_likelihood_ratio = math.log((count_with_sense + self.smoothing) / (count_without_sense + self.smoothing))
            if log_likelihood_ratio > 0:
                self.rules[feature] = (log_likelihood_ratio, sense)
        return self

    def ranked_rules(self):
        return sorted(((score, feature, sense) for feature, (score, sense) in self.rules.items()), reverse=True)

    def predict(self, features):
        matching_rules = [self.rules[feature] for feature in features if feature in self.rules]
        if not matching_rules:
            return self.most_frequent_sense
        return max(matching_rules)[1]
