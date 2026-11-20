import math

import numpy as np
from scipy import sparse

from ..folds import split_off_development_documents
from .base import SenseClassifier


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class LogisticRegression(SenseClassifier):
    """Multinomial logistic regression trained from scratch with mini-batch SGD and L2."""

    def __init__(self, learning_rate=0.5, l2_strength=1e-4, epochs=20, batch_size=16, seed=0):
        self.learning_rate = learning_rate
        self.l2_strength = l2_strength
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def to_matrix(self, feature_lists):
        rows, columns = [], []
        for row, features in enumerate(feature_lists):
            for column in {self.feature_index[f] for f in features if f in self.feature_index}:
                rows.append(row)
                columns.append(column)
        values = np.ones(len(rows), dtype=np.float64)
        return sparse.csr_matrix((values, (rows, columns)), shape=(len(feature_lists), len(self.feature_index)))

    def fit(self, feature_lists, senses, after_epoch=None):
        """Train; `after_epoch(epoch_number, model)` is called after every epoch if given."""
        self.senses = sorted(set(senses))
        sense_index = {sense: i for i, sense in enumerate(self.senses)}
        all_features = sorted({feature for features in feature_lists for feature in features})
        self.feature_index = {feature: i for i, feature in enumerate(all_features)}

        inputs = self.to_matrix(feature_lists)
        targets = np.zeros((len(senses), len(self.senses)))
        targets[np.arange(len(senses)), [sense_index[sense] for sense in senses]] = 1.0

        self.weights = np.zeros((len(self.feature_index), len(self.senses)))
        self.bias = np.zeros(len(self.senses))
        random_generator = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            step_size = self.learning_rate / np.sqrt(1 + epoch)
            order = random_generator.permutation(len(senses))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                batch_inputs = inputs[batch]
                probabilities = softmax(batch_inputs @ self.weights + self.bias)
                error = (probabilities - targets[batch]) / len(batch)
                self.weights -= step_size * (np.asarray(batch_inputs.T @ error) + self.l2_strength * self.weights)
                self.bias -= step_size * error.sum(axis=0)
            if after_epoch:
                after_epoch(epoch + 1, self)
        return self

    def predict_distributions(self, feature_lists):
        probabilities = softmax(self.to_matrix(feature_lists) @ self.weights + self.bias)
        return [dict(zip(self.senses, row.tolist())) for row in probabilities]

    def predict_distribution(self, features):
        return self.predict_distributions([features])[0]


def tune_logistic_regression(feature_lists, senses, documents, l2_grid, epoch_checkpoints, seed=0,
                             default=(1e-4, 20)):
    """Choose L2 strength and number of epochs by accuracy (log-loss breaks ties) on held-out documents."""
    development_documents = split_off_development_documents(documents)
    train_rows = [i for i, document in enumerate(documents) if document not in development_documents]
    development_rows = [i for i, document in enumerate(documents) if document in development_documents]
    if not train_rows or not development_rows:
        return default

    development_features = [feature_lists[i] for i in development_rows]
    development_senses = [senses[i] for i in development_rows]
    results = {}

    for l2_strength in l2_grid:
        def record(epoch, model):
            if epoch not in epoch_checkpoints:
                return
            distributions = model.predict_distributions(development_features)
            correct = sum(max(d, key=d.get) == gold for d, gold in zip(distributions, development_senses))
            log_loss = -sum(math.log(max(d.get(gold, 0.0), 1e-12)) for d, gold in zip(distributions, development_senses))
            results[(l2_strength, epoch)] = (correct, -log_loss)

        LogisticRegression(l2_strength=l2_strength, epochs=max(epoch_checkpoints), seed=seed).fit(
            [feature_lists[i] for i in train_rows], [senses[i] for i in train_rows], after_epoch=record)

    return max(results, key=results.get)
