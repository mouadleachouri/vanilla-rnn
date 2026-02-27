"""Main RNN model logic."""

import numpy as np

from helpers import (
    Gradients,
    HyperParams,
    Params,
    bin_cross_entropy,
    init_params,
    sigmoid,
)


def rnn_cell_forward(
    a_prev: np.ndarray,
    x: np.ndarray,
    params: Params,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """RNN one step forward.

    Parameters
    ----------
    a_prev : numpy.ndarray
        Previous batch of activations. Of shape `(state_size, batch_size)`.
    x : numpy.ndarray
        Current batch of input embeddings. Of shape `(vocab_size, batch_size)`.
    params : dict
        Dictionary of trainable paramters. Including:
        * `W_ax`: of shape `(state_size, vocab_size)`.
        * `W_aa`: of shape `(state_size, state_size)`.
        * `b_a`: of shape `(state_size, 1)`.
        * `W_y`: of shape `(1, state_size)`.
        * `b_y`: of shape `(1, 1)`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, dict of str -> numpy.ndarray]
        * The current batch of activations. Of shape `(state_size, batch_size)`.
        * The current batch of outputs. Of shape `(1, batch_size)`.
        * The cache used by backpropagation, containing:
            * `a`: of shape `(state_size, batch_size)`.
            * `a_prev`: of shape `(state_size, batch_size)`.
            * `y_hat`: of shape `(1, batch_size)`.

    """
    a = np.tanh(params["W_aa"] @ a_prev + params["W_ax"] @ x + params["b_a"])
    y = sigmoid(params["W_y"] @ a + params["b_y"])
    cache = (a, a_prev, y)
    return a, y, cache


def rnn_forward(
    a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    params: Params,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """RNN forward pass.

    Parameters
    ----------
    a : numpy.ndarray
        Batch of initial activations. Of shape `(state_size, batch_size)`.
    x : numpy.ndarray
        A batch of full sequences of embeddings. Of shape
        `(vocab_size, batch_size, seq_length)`.
    y : numpy.ndarray
        The batch of ground truth outputs for x. Of shape `(1, batch_size, seq_length)`.
    params : Params
        Dictionary of trainable parameters. Including:
        * `W_ax`: of shape `(state_size, vocab_size)`.
        * `W_aa`: of shape `(state_size, state_size)`.
        * `b_a`: of shape `(state_size, 1)`.
        * `W_y`: of shape `(1, state_size)`.
        * `b_y`: of shape `(1, 1)`.

    Returns
    -------
    tuple[numpy.ndarray, list[tuple[str, np.ndarray]]]
        * The binary cross entropy loss. Of shape `(1, batch_size)`.
        * The caches used by backward propagation

    """
    _, batch_size, seq_length = x.shape
    loss = np.zeros(shape=(1, batch_size))
    caches = []
    for t in range(seq_length):
        x_t, y_t = x[:, :, t], y[:, :, t]
        a, y_hat, cache = rnn_cell_forward(a, x_t, params)
        caches.append(cache)
        loss += bin_cross_entropy(y_t, y_hat)
    return loss.sum() / (batch_size * seq_length), caches


def rnn_backward(
    dz_a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    caches: list[tuple],
    params: Params,
) -> Gradients:
    """Compute the gradients of the loss wrt trainable parameters.

    Parameters
    ----------
    dz_a : numpy.ndarray
        Initial gradient of the loss wrt `z_a` (activation `a` before `tanh` is
        applied). Should be set to zeros. Of shape `(state_size, batch_size)`.
    x : numpy.ndarray
        A batch of full sequences of embeddings. Of shape
        `(vocab_size, batch_size, seq_length)`.
    y : numpy.ndarray
        The batch of ground truth outputs for x. Of shape
        `(1, batch_size, seq_length)`.
    caches : list of tuple
        The cached values needed for computing the gradients.
    params : Params
        Dictionary of trainable parameters. Including:
        * `W_ax`: of shape `(state_size, vocab_size)`.
        * `W_aa`: of shape `(state_size, state_size)`.
        * `b_a`: of shape `(state_size, 1)`.
        * `W_y`: of shape `(1, state_size)`.
        * `b_y`: of shape `(1, 1)`.

    """
    _, batch_size, seq_length = y.shape

    # Initialize gradients
    dW_y = np.zeros_like(params["W_y"])  # noqa: N806
    db_y = np.zeros_like(params["b_y"])
    dW_ax = np.zeros_like(params["W_ax"])  # noqa: N806
    dW_aa = np.zeros_like(params["W_aa"])  # noqa: N806
    db_a = np.zeros_like(params["b_a"])

    for t in range(seq_length - 1, -1, -1):
        x_t, y_t = x[:, :, t], y[:, :, t]
        a, a_prev, y_hat = caches[t]
        dy = (y_hat - y_t) / (batch_size * seq_length)
        dz_a = (params["W_y"].T @ dy + params["W_aa"].T @ dz_a) * (1 - a**2)

        # Increment gradients
        dW_y += dy @ a.T  # noqa: N806
        db_y += np.sum(dy, axis=1, keepdims=True)
        dW_ax += dz_a @ x_t.T  # noqa: N806
        dW_aa += dz_a @ a_prev.T  # noqa: N806
        db_a += np.sum(dz_a, axis=1, keepdims=True)

    # Return clipped gradients
    return {
        "dW_ax": np.clip(dW_ax, -5, 5),
        "dW_aa": np.clip(dW_aa, -5, 5),
        "db_a": np.clip(db_a, -5, 5),
        "dW_y": np.clip(dW_y, -5, 5),
        "db_y": np.clip(db_y, -5, 5),
    }


def train_step(  # noqa: PLR0913
    a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dz_a: np.ndarray,
    params: Params,
    hyperparams: HyperParams,
) -> np.ndarray:
    """One step of the training process.

    A full forward and backward BPTT pass.

    Parameters
    ----------
    a : numpy.ndarray
        Batch of initial activations. Of shape `(state_size, batch_size)`.
    x : numpy.ndarray
        A batch of full sequences of embeddings. Of shape
        `(vocab_size, batch_size, seq_length)`.
    y : numpy.ndarray
        The batch of ground truth outputs for x. Of shape `(1, batch_size, seq_length)`.
    dz_a : numpy.ndarray
        Initial gradient of the loss wrt `z_a` (activation `a` before `tanh` is
        applied). Should be set to zeros. Of shape `(state_size, batch_size)`.
    params : Params
        Dictionary of trainable parameters. Including:
        * `W_ax`: of shape `(state_size, vocab_size)`.
        * `W_aa`: of shape `(state_size, state_size)`.
        * `b_a`: of shape `(state_size, 1)`.
        * `W_y`: of shape `(1, state_size)`.
        * `b_y`: of shape `(1, 1)`.
    hyperparams : HyperParams
        Dictionary of hyperparameters. Including:
        * `learning_rate`: learning rate.

    Returns
    -------
    numpy.ndarray
        The batch of losses, of shape `(1, batch_size)`.

    """
    loss, caches = rnn_forward(a, x, y, params)
    grads = rnn_backward(dz_a, x, y, caches, params)
    params["W_ax"] -= hyperparams["learning_rate"] * grads["dW_ax"]
    params["W_aa"] -= hyperparams["learning_rate"] * grads["dW_aa"]
    params["b_a"] -= hyperparams["learning_rate"] * grads["db_a"]
    params["W_y"] -= hyperparams["learning_rate"] * grads["dW_y"]
    params["b_y"] -= hyperparams["learning_rate"] * grads["db_y"]
    return loss


def train(
    x: np.ndarray,
    y: np.ndarray,
    hyperparams: HyperParams,
    seed: int,
    *,
    verbose: bool = True,
) -> list[float]:
    """Train the RNN model.

    Parameters
    ----------
    x : numpy.ndarray
        A batch of full sequences of embeddings. Of shape
        `(vocab_size, batch_size, seq_length)`.
    y : numpy.ndarray
        The batch of ground truth outputs for x. Of shape `(1, batch_size, seq_length)`.
    hyperparams : HyperParams
        Dictionary of hyperparameters. Including:
        * `learning_rate`: learning rate.
    seed : int
        Seed for numpy.random (for weight initialization).
    verbose : bool
        Wether to print the loss for each epoch.

    Returns
    -------
    list of float
        The means of losses across all batches, for every epoch.

    """
    batch_size = hyperparams["batch_size"]
    state_size = hyperparams["state_size"]
    x, y = (
        x[:, : (x.shape[1] // batch_size) * batch_size, :],
        y[:, : (y.shape[1] // batch_size) * batch_size, :],
    )
    vocab_size, num_examples, _ = x.shape
    params = init_params(vocab_size, state_size, seed)
    loss_per_epoch = []
    for epoch in range(hyperparams["epochs"]):
        losses = []
        for i in range(0, num_examples, batch_size):
            x_batch, y_batch = x[:, i : i + batch_size, :], y[:, i : i + batch_size, :]
            a = np.zeros(shape=(state_size, batch_size))
            dz_a = np.zeros(shape=(state_size, batch_size))
            batch_loss = train_step(a, x_batch, y_batch, dz_a, params, hyperparams)
            losses.append(batch_loss.mean())
        global_loss = np.array(losses).mean()
        loss_per_epoch.append(global_loss)
        if verbose:
            print(  # noqa: T201
                f"Epoch {epoch + 1:03}/{hyperparams['epochs']:03} - "
                f"Loss: {global_loss:.2f}",
            )
    return loss_per_epoch


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(scale=1, size=(100, 1000, 10))
    y = rng.integers(0, 2, size=(1, 1000, 10))
    hyperparams: HyperParams = {
        "learning_rate": 0.005,
        "batch_size": 32,
        "state_size": 32,
        "epochs": 10000,
    }
    import matplotlib.pyplot as plt

    plt.plot(train(x, y, hyperparams, 42))
    plt.show()
