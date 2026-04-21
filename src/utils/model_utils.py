import numpy as np
import pickle
from pathlib import Path


class RidgeRegressor:
    """Ridge Regression using closed-form solution."""

    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg
        self.W = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y, standardize=True):
        """Fit Ridge Regression.

        Args:
            X: (N, d) feature matrix
            y: (N, m) target matrix
            standardize: whether to standardize features

        Returns:
            self
        """
        if standardize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-7
            X_std = (X - self.mean_) / self.std_
        else:
            X_std = X

        # Closed-form: W = (X^T X + λI)^-1 X^T y
        XtX = X_std.T @ X_std
        Xty = X_std.T @ y
        d = X_std.shape[1]
        self.W = np.linalg.solve(XtX + self.lambda_reg * np.eye(d), Xty)
        return self

    def predict(self, X):
        """Predict using fitted model.

        Args:
            X: (N, d) feature matrix

        Returns:
            (N, m) predictions, clipped to [0, 1]
        """
        if self.mean_ is not None:
            X_std = (X - self.mean_) / self.std_
        else:
            X_std = X

        y_pred = X_std @ self.W
        return np.clip(y_pred, 0.0, 1.0)

    def save(self, path):
        """Save model to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({'W': self.W, 'lambda': self.lambda_reg,
                        'mean': self.mean_, 'std': self.std_}, f)

    @classmethod
    def load(cls, path):
        """Load model from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(lambda_reg=data['lambda'])
        model.W = data['W']
        model.mean_ = data['mean']
        model.std_ = data['std']
        return model


def batch_ridges(X_train, y_train, X_val, y_val, X_test, y_test,
                 lambda_grid, audio_features):
    """Evaluate Ridge for multiple lambda values.

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        lambda_grid: array of lambda values to test
        audio_features: list of audio feature names

    Returns:
        list: results dicts
    """
    results = []

    for lam in lambda_grid:
        model = RidgeRegressor(lambda_reg=lam)
        model.fit(X_train, y_train, standardize=True)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        results.append({
            'lambda': lam,
            'train_mse': float(np.mean((y_train - y_train_pred) ** 2)),
            'val_mse': float(np.mean((y_val - y_val_pred) ** 2)),
            'test_mse': float(np.mean((y_test - y_test_pred) ** 2)),
        })

    return results


class NeuralNetworkRegressor:
    """Two-layer neural network for multi-output regression (from scratch)."""

    def __init__(self, hidden_sizes=(128, 64), learning_rate=0.005,
                 batch_size=32, dropout_rate=0.0, l2_lambda=0.0):
        """Initialize ANN.

        Args:
            hidden_sizes: tuple of hidden layer sizes, e.g., (128, 64)
            learning_rate: gradient descent step size
            batch_size: samples per batch
            dropout_rate: dropout probability (0.0-1.0)
            l2_lambda: L2 regularization strength
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.mean_ = None
        self.std_ = None
        self.weights = []
        self.biases = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _initialize_weights(self, input_dim, output_dim):
        """Initialize weights with small random values."""
        self.weights = []
        self.biases = []

        layer_sizes = [input_dim] + list(self.hidden_sizes) + [output_dim]

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def fit(self, X, y, X_val=None, y_val=None, epochs=500,
            early_stopping_patience=20, standardize=True, verbose=False):
        """Train the neural network.

        Args:
            X: (N, d) training features
            y: (N, m) training targets
            X_val: (N_val, d) validation features (for early stopping)
            y_val: (N_val, m) validation targets
            epochs: max training epochs
            early_stopping_patience: stop if val loss doesn't improve for N epochs
            standardize: whether to standardize features
            verbose: print progress

        Returns:
            self
        """
        if standardize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-7
            X_train = (X - self.mean_) / self.std_
        else:
            X_train = X.copy()

        self._initialize_weights(X_train.shape[1], y.shape[1])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            for start_idx in range(0, len(X_train), self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y[batch_idx]

                # Forward pass
                z_list, a_list = self._forward_pass(X_batch, training=True)

                # Backward pass
                self._backward_pass(X_batch, y_batch, z_list, a_list)

            # Validation check for early stopping
            if X_val is not None and y_val is not None:
                y_val_std = (X_val - self.mean_) / self.std_ if standardize else X_val
                _, val_pred = self._forward_pass(y_val_std, training=False)
                val_loss = float(np.mean((y_val - val_pred[-1]) ** 2))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return self

    def _forward_pass(self, X, training=False):
        """Forward pass through network.

        Returns:
            z_list: pre-activation values for each layer
            a_list: post-activation values for each layer
        """
        z_list = []
        a_list = [X]

        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            z_list.append(z)

            a = self._relu(z)

            # Apply dropout during training
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape)
                a = a * mask / (1 - self.dropout_rate)

            a_list.append(a)
            current = a

        # Output layer: sigmoid for [0, 1] range
        z_output = current @ self.weights[-1] + self.biases[-1]
        z_list.append(z_output)
        a_output = self._sigmoid(z_output)
        a_list.append(a_output)

        return z_list, a_list

    def _backward_pass(self, X, y, z_list, a_list):
        """Backward pass: compute gradients and update weights."""
        m = len(X)
        delta = (a_list[-1] - y) / m

        # Output layer gradient
        dW = a_list[-2].T @ delta + 2 * self.l2_lambda * self.weights[-1]
        db = np.sum(delta, axis=0, keepdims=True)

        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db

        # Hidden layers (ReLU)
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self._relu_grad(z_list[i])

            dW = a_list[i].T @ delta + 2 * self.l2_lambda * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True)

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def predict(self, X):
        """Predict on new data.

        Args:
            X: (N, d) features

        Returns:
            (N, m) predictions, clipped to [0, 1]
        """
        if self.mean_ is not None:
            X_std = (X - self.mean_) / self.std_
        else:
            X_std = X

        _, a_list = self._forward_pass(X_std, training=False)
        return np.clip(a_list[-1], 0.0, 1.0)

    def save(self, path):
        """Save model to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'hidden_sizes': self.hidden_sizes,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate,
                'l2_lambda': self.l2_lambda,
                'mean': self.mean_,
                'std': self.std_,
            }, f)

    @classmethod
    def load(cls, path):
        """Load model from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            hidden_sizes=data['hidden_sizes'],
            learning_rate=data['learning_rate'],
            dropout_rate=data['dropout_rate'],
            l2_lambda=data['l2_lambda']
        )
        model.weights = data['weights']
        model.biases = data['biases']
        model.mean_ = data['mean']
        model.std_ = data['std']
        return model


def compute_metrics(y_true, y_pred, genre_centroids, genre_names, audio_features):
    """Compute all evaluation metrics.

    Args:
        y_true: (N, 7) true audio profiles
        y_pred: (N, 7) predicted audio profiles
        genre_centroids: (G, 7) genre centroids
        genre_names: list of G genre names
        audio_features: list of 7 audio feature names

    Returns:
        dict: all metrics
    """
    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # Cosine Similarity
    dot = np.sum(y_true * y_pred, axis=1)
    norm_t = np.linalg.norm(y_true, axis=1)
    norm_p = np.linalg.norm(y_pred, axis=1)
    denom = norm_t * norm_p
    denom = np.where(denom == 0, 1.0, denom)
    cos_sim = float(np.mean(dot / denom))

    # R2 Score (per-dimension, then averaged)
    r2_scores = []
    for j in range(y_true.shape[1]):
        tss = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)
        rss = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
        r2_scores.append(r2)
    r2 = float(np.mean(r2_scores))

    # Top-K Accuracy (COMMENTED OUT per requirements - preserved for reference)
    # def top_k_acc(k=3):
    #     hits = 0
    #     for i in range(len(y_true)):
    #         dist_true = np.linalg.norm(genre_centroids - y_true[i], axis=1)
    #         target_genre = genre_names[np.argmin(dist_true)]
    #
    #         dist_pred = np.linalg.norm(genre_centroids - y_pred[i], axis=1)
    #         top_k_genres = np.array(genre_names)[np.argsort(dist_pred)[:k]]
    #
    #         if target_genre in top_k_genres:
    #             hits += 1
    #     return hits / len(y_true)
    #
    # top3 = top_k_acc(k=3)
    # top5 = top_k_acc(k=5)

    # Placeholder values for backward compatibility
    top3 = 0.0
    top5 = 0.0

    # Per-dimension MSE
    per_dim_mse = {feat: float(np.mean((y_true[:, j] - y_pred[:, j]) ** 2))
                   for j, feat in enumerate(audio_features)}

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'cos_sim': cos_sim,
        'top3_acc': top3,  # COMMENTED OUT - returns 0.0 for backward compat
        'top5_acc': top5,  # COMMENTED OUT - returns 0.0 for backward compat
        'per_dimension_mse': per_dim_mse,
    }
