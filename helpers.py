"""Helpers for the RNN model."""

from typing import TypedDict

import numpy as np


class HyperParams(TypedDict):
    """Model hyperparameters."""

    learning_rate: float
    state_size: int
    batch_size: int
    epochs: int


class Params(TypedDict):
    """Model trainable parameters."""

    W_ax: np.ndarray
    W_aa: np.ndarray
    b_a: np.ndarray
    W_y: np.ndarray
    b_y: np.ndarray


class Gradients(TypedDict):
    """Loss gradients wrt trainable parameters."""

    dW_ax: np.ndarray
    dW_aa: np.ndarray
    db_a: np.ndarray
    dW_y: np.ndarray
    db_y: np.ndarray


def sigmoid(
    x: np.ndarray,
) -> np.ndarray:
    """Compute the sigmoid of x.

    Parameters
    ----------
    x : numpy.ndarray
        A vector of real number.

    Returns
    -------
    numpy.ndarray
        The sigmoid function, evaluated at `x`, element-wise.

    """
    return 1 / (1 + np.exp(-x))


def bin_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the binary cross entropy loss.

    Parameters
    ----------
    y_true : numpy.ndarray
        The batch of true labels. Of shape `(1, batch_size)`.
    y_pred : numpy.ndarray
        The batch of pred labels. Of shape `(1, batch_size)`.

    Returns
    -------
    numpy.ndarray
        The batch of corresponding losses.

    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def init_params(
    vocab_size: int,
    state_size: int,
    seed: int | None = None,
) -> Params:
    """Initialize RNN parameters.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (and of the input embeddings).
    state_size : int
        Size of the RNN hidden state.
    seed : int
        Seed for the numpy rng.

    Returns
    -------
    dict of str -> numpy.ndarray
        Dictionary of parameters, including:
        * `W_ax`: of shape `(state_size, vocab_size)`, initialized
        randomly following a normal distribution.
        * `W_aa`: of shape `(state_size, state_size)`, initialized
        randomly following a normal distribution.
        * `b_a`: of shape `(state_size, 1)`, initialized with zeros.
        * `W_y`: of shape `(1, state_size)`, initialized
        randomly following a normal distribution.
        * `b_y`: of shape `(1, 1)`, initialized with zeros.

    """
    rng = np.random.default_rng(seed)
    return {
        "W_ax": rng.normal(scale=0.01, size=(state_size, vocab_size)),
        "W_aa": rng.normal(scale=0.01, size=(state_size, state_size)),
        "b_a": np.zeros(shape=(state_size, 1)),
        "W_y": rng.normal(scale=0.01, size=(1, state_size)),
        "b_y": np.zeros(shape=(1, 1)),
    }
