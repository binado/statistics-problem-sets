import numpy as np

def gaussian(x, mu: float, sigma: float):
    norm = np.sqrt(2 * np.pi) * sigma
    chi_squared = (x - mu) ** 2 / sigma ** 2
    return np.exp(-0.5 * chi_squared) / norm

def loggaussian(x, mu: float, sigma: float):
    norm = np.sqrt(2 * np.pi) * sigma
    chi_squared = (x - mu) ** 2 / sigma ** 2
    return -0.5 * chi_squared - np.log(norm)

def uniform_prior(x, a, b):
    return 1 / (b - a)

def jeffreys_prior(x, a, b):
    return 1 / x / np.log(b / a)

def safe_exp(x):
    xmax = np.max(x)
    return np.exp(xmax) * np.exp(x - xmax)

def normalize(y, x):
    return y / np.trapz(y, x)
