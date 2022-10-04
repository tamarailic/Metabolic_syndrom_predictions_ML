import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, gamma=0.001, iterations=100, omega1=0.9, omega2=0.99, epsilon1=1e-6, epsilon=1e-6):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0
        v_w = np.ones(shape=self.weights.shape)
        m_w = np.ones(shape=self.weights.shape)
        v_b = 1
        m_b = 1

        for i in range(iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (2 / samples) * np.dot(X.T, (y_pred - y))
            db = (2 / samples) * np.sum(y_pred - y)

            m_w = omega1 * m_w + (1 - omega1) * dw
            v_w = omega2 * v_w + (1 - omega2) * np.multiply(dw, dw)
            hat_v_w = np.abs(v_w / (1 - omega2))
            hat_m_w = m_w / (1 - omega1)
            self.weights = self.weights - gamma * np.ones(shape=dw.shape) / np.sqrt(hat_v_w + epsilon1) * hat_m_w

            m_b = omega1 * m_b + (1 - omega1) * db
            v_b = omega2 * v_b + (1 - omega2) * db ** 2
            hat_v_b = abs(v_b / (1 - omega2))
            hat_m_b = m_b / (1 - omega1)
            self.bias = self.bias - gamma / np.sqrt(hat_v_b + epsilon1) * hat_m_b

            if np.linalg.norm(dw) < epsilon or db < epsilon:
                break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))