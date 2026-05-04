"""Product / isotropic Matérn-1/2 kernels."""
import numpy as np

KERNEL_L = 1.0


def _matern12_product(X: np.ndarray, Y: np.ndarray, L: float = KERNEL_L) -> np.ndarray:
    K = np.ones((X.shape[0], Y.shape[0]))
    for i in range(X.shape[1]):
        K *= np.exp(-np.abs(X[:, i:i+1] - Y[np.newaxis, :, i]) / L)
    return K


def _matern12_isotropic(X: np.ndarray, Y: np.ndarray, L: float = KERNEL_L) -> np.ndarray:
    """Isotropic Matérn-1/2 (exponential) kernel: ``exp(-‖x-x'‖_2 / L)``."""
    diff = X[:, None, :] - Y[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.exp(-dist / L)

