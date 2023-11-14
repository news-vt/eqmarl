import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray, axis: int = None):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)