import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight


class DataWagonModel:
    """
    :param cat_features: list of cat features name
    :param class_weight: dict
    :param iterations: number of boosting iterations
    :param depth: trees depth
    """

    def __init__(self, cat_features, class_weight, iterations, depth):
        self.cat_features = cat_features
        self.class_weight = class_weight
        self.iterations = iterations
        self.depth = depth
        self.model = None

    def fit(self, X, y):
        """
        :param X: features df
        :param y: target series
        """
        classes = np.unique(y.values)
        weights = compute_class_weight(
            y=y.values,
            classes=classes,
            class_weight=self.class_weight
        )
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            cat_features=self.cat_features,
            class_weights=weights,
            random_state=102,
            verbose=10
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        :param X: features df
        :return: probability of class 1
        """
        if self.model is None:
            raise Exception('Model is not fitted yet.')
        predictions = self.model.predict_proba(X)
        return predictions

    def predict(self, X):
        """
        :param X: features df
        :return: class number
        """
        predictions = self.predict_proba(X)
        return (predictions > 0.5).astype(int)
