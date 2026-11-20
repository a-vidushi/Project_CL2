class SenseClassifier:
    """Common interface: train on feature lists, predict a sense (and a distribution when the model has one)."""

    def fit(self, feature_lists, senses):
        raise NotImplementedError

    def predict_distribution(self, features):
        """Return {sense: probability}, or None if the model is not probabilistic."""
        return None

    def predict(self, features):
        distribution = self.predict_distribution(features)
        return max(distribution, key=distribution.get)
