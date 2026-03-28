"""Shared neural network primitives for signalfault.classify modules.

Low-level building blocks: activation, normalization, and N-layer sigmoid
NN training with Huber loss, momentum SGD, He initialization, and gradient
clipping.  Used by nhanes, superconductor, and metallic_glass cascades.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation & normalization
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Vectorized sigmoid with clipping for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_norm_stats(X):
    """Per-column z-score normalization statistics.

    Returns:
        (means, stds) where stds are floored at 1e-12
    """
    means = X.mean(axis=0)
    stds = np.maximum(X.std(axis=0), 1e-12)
    return means, stds


def normalize(X, means, stds):
    """Z-score normalize."""
    return (X - means) / stds


# ---------------------------------------------------------------------------
# N-layer sigmoid NN with Huber loss
# ---------------------------------------------------------------------------

def train_nl(Xn, y, hidden_sizes, n_epochs, seed, lr=0.01, l2=0.01,
             huber_delta=5.0, clip_grad=5.0, momentum=0.9):
    """Train an N-layer sigmoid NN with Huber loss, momentum SGD.

    Architecture: input -> [sigmoid hidden]* -> linear scalar output.
    Uses He initialization for weights.

    Args:
        Xn: (n, d) normalized feature matrix
        y: (n,) float64 target
        hidden_sizes: list of ints, hidden layer widths
        n_epochs: number of training epochs
        seed: random seed for weight init
        lr: learning rate
        l2: L2 regularization coefficient
        huber_delta: Huber loss clipping threshold
        clip_grad: gradient clipping threshold
        momentum: momentum coefficient

    Returns:
        dict with keys 'W' (list of weight matrices), 'b' (list of bias vectors)
    """
    rng = np.random.RandomState(seed)
    n, d = Xn.shape
    sizes = [d] + list(hidden_sizes) + [1]
    W = [rng.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / sizes[i])
         for i in range(len(sizes) - 1)]
    b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    vW = [np.zeros_like(w) for w in W]
    vb = [np.zeros_like(bi) for bi in b]

    for ep in range(n_epochs):
        # Forward
        activations = [Xn]
        for i in range(len(W)):
            z = activations[-1] @ W[i] + b[i]
            activations.append(sigmoid(z) if i < len(W) - 1 else z.ravel())

        # Huber gradient
        err = activations[-1] - y
        huber_grad = np.where(np.abs(err) <= huber_delta, err,
                              huber_delta * np.sign(err))
        grad = huber_grad.reshape(-1, 1) / n

        # Backprop with gradient clipping and momentum
        for i in range(len(W) - 1, -1, -1):
            dW = activations[i].T @ grad + l2 * W[i]
            db = grad.mean(axis=0)
            np.clip(dW, -clip_grad, clip_grad, out=dW)
            np.clip(db, -clip_grad, clip_grad, out=db)
            if i > 0:
                grad = (grad @ W[i].T) * activations[i] * (1 - activations[i])
            vW[i] = momentum * vW[i] - lr * dW
            W[i] += vW[i]
            vb[i] = momentum * vb[i] - lr * db
            b[i] += vb[i]

    return {'W': W, 'b': b}


def predict_nl(Xn, params):
    """Forward pass through a trained N-layer sigmoid NN.

    Args:
        Xn: (n, d) normalized feature matrix
        params: dict with 'W' (list), 'b' (list)

    Returns:
        (n,) float64 predictions
    """
    a = Xn
    for i in range(len(params['W'])):
        z = a @ params['W'][i] + params['b'][i]
        a = sigmoid(z) if i < len(params['W']) - 1 else z.ravel()
    return a
