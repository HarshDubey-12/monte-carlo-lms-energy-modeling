import numpy as np

class LMSRegressor:
    def __init__(self, n_features, learning_rate=0.01):
        self.lr = learning_rate
        self.w = np.zeros(n_features)
        self.b = 0.0
        # Tracking for diagnostics
        self.loss_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def update(self, x, y):
        """
        Perform one LMS update using a single sample
        """
        y_hat = np.dot(self.w, x) + self.b
        error = y - y_hat

        # LMS Update
        self.w += self.lr * error * x
        self.b += self.lr * error

        return error

    def fit(self, X, y, n_epochs, shuffle=True):
        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            epoch_loss = 0.0

            for i in indices:
                error = self.update(X[i], y[i])
                epoch_loss += 0.5 * error ** 2
            self.loss_history.append(epoch_loss / n_samples)