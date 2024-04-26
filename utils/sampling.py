import numpy as np
from scipy.integrate import cumulative_trapezoid

def inversion_sampling(pdf: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Return samples from a function with known PDF using the inversion technique.

    Args:
        pdf (np.ndarray): Distribution's pdf. Does not need to be normalized
        x (np.ndarray): Domain of integration. Samples will be drawn from this array
        n (int): Number of samples to return

    Returns:
        np.ndarray: n samples from pdf
    """
    cdf = cumulative_trapezoid(pdf, x)
    max_cdf = cdf[-1]
    percentiles = np.random.uniform(high=max_cdf, size=n)
    return x[np.searchsorted(cdf, percentiles)]
