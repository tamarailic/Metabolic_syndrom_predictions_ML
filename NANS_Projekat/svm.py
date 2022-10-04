import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y, omega1=0.9, omega2=0.99, epsilon1=1e-6, epsilon=1e-6):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        #         for _ in range(self.n_iters):
        #             for idx, x_i in enumerate(X):
        #                 condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
        #                 if condition:
        #                     self.w -= self.lr * (2 * self.lambda_param * self.w)
        #                 else:
        #                     self.w -= self.lr * (
        #                         2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
        #                     )
        #                     self.b -= self.lr * y_[idx]

        v_w = np.ones(shape=self.w.shape)
        m_w = np.ones(shape=self.w.shape)
        v_b = 1
        m_b = 1

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0

                    m_w = omega1 * m_w + (1 - omega1) * dw
                    v_w = omega2 * v_w + (1 - omega2) * np.multiply(dw, dw)
                    hat_v_w = np.abs(v_w / (1 - omega2))
                    hat_m_w = m_w / (1 - omega1)
                    self.w -= self.lr * np.ones(shape=dw.shape) / np.sqrt(hat_v_w + epsilon1) * hat_m_w

                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]

                    m_w = omega1 * m_w + (1 - omega1) * dw
                    v_w = omega2 * v_w + (1 - omega2) * np.multiply(dw, dw)
                    hat_v_w = np.abs(v_w / (1 - omega2))
                    hat_m_w = m_w / (1 - omega1)
                    self.w -= self.lr * np.ones(shape=dw.shape) / np.sqrt(hat_v_w + epsilon1) * hat_m_w

                    m_b = omega1 * m_b + (1 - omega1) * db
                    v_b = omega2 * v_b + (1 - omega2) * db ** 2
                    hat_v_b = abs(v_b / (1 - omega2))
                    hat_m_b = m_b / (1 - omega1)
                    self.b -= self.lr / np.sqrt(hat_v_b + epsilon1) * hat_m_b

                if np.linalg.norm(dw) < epsilon:
                    break

    def predict(self, X):

        approx = np.dot(X, self.w) - self.b
        return np.where(np.sign(approx) == -1,0,1)