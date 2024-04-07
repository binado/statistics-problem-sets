import numpy as np

def gaussian(x: np.ndarray, mu: float, sigma: float):
    norm = 1. / np.sqrt(2 * np.pi) * sigma
    chi_squared = (x - mu) ** 2 / sigma ** 2
    return norm * np.exp(-0.5 * chi_squared)
