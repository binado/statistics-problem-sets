import numpy as np
from scipy.integrate import cumulative_trapezoid

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

def quantile(pdf, x, qs, discrete=False):
    cdf = np.cumsum(pdf) if discrete else cumulative_trapezoid(pdf, x=x)
    return x[np.searchsorted(cdf, qs)]

def hdi(pdf, x, alpha):
    """
    Compute the alpha-Highest Density Interval (HDI) of a given pdf along the array x.
    """
    # Use cdf: monotonic in [0,1]
    cdf = cumulative_trapezoid(pdf, x=x)
    # Get index i for which cdf[i] = alpha
    median, l = np.searchsorted(cdf, [0.5, alpha])
    n = len(cdf)
    # Minimal length of interval
    minl = l
    # Index for low-end of minimal-length interval
    mini = 0
    maxalpha = cdf[mini + minl] - cdf[minl]
    for i in range(1, n):
        # At the start of the loop i is increased by 1
        # but high-end point is kept fixed, therefore decrease l by 1
        l = minl - 1
        alphai = cdf[i + l] - cdf[i]
        # Increase l until the interval [i, i + l] encloses the desired CI
        while alphai < alpha and l <= minl:
            l += 1
            if i + l >= n:
                # If this condition is met, i is already too high
                # and we can terminate
                # Return median and hdi values in x
                return x[median], x[mini], x[mini + minl]

            alphai = cdf[i + l] - cdf[i]
            if l > minl:
                # Not worth increasing interval if it already non-maximal
                break
        if l < minl or l == minl and alphai > maxalpha:
            minl = l
            mini = i
            maxalpha = alphai
