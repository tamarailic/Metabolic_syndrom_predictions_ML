import numpy as np
from rf import most_common_label

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self.predicted_label(x) for x in X]
        return np.array(predicted_labels)

    def predicted_label(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return most_common_label(k_nearest_labels)