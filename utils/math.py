import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma
from scipy.fft import rfft, irfft, next_fast_len

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

def normalize(y: np.ndarray, x=None, discrete=False) -> np.ndarray:
    if not discrete and x is None:
        raise ValueError

    norm = np.sum(y) if discrete else np.trapz(y, x)
    return y / norm

def quantile(pdf: np.ndarray, x: np.ndarray, qs, discrete=False) -> np.ndarray:
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
                return median, mini, mini + minl

            alphai = cdf[i + l] - cdf[i]
            if l > minl:
                # Not worth increasing interval if it already non-maximal
                break
        if l < minl or l == minl and alphai > maxalpha:
            minl = l
            mini = i
            maxalpha = alphai


def moment(pdf: np.ndarray, x: np.ndarray, n: int, central=True) -> float:
    """Calculate nth order moment of a given pdf along x.

    Args:
        pdf (np.npdarray): pdf valued at x
        x (np.ndarray): domain of integration
        n (int): moment order
        central (bool, optional): compute central moment. Defaults to True.

    Returns:
        float: nth order moment
    """  
    mu = moment(pdf, x, 1) if central and n > 1 else 0.
    return np.trapz(pdf * (x - mu) ** n, x)


def beta_pdf(x: np.ndarray, alpha: float, beta: float):
    domain = (0 < x) & (x < 1)
    norm = gamma(alpha + beta) / gamma(alpha) / gamma(beta)
    return domain * norm * x ** (alpha - 1) * (1 - x) ** (beta - 1)

def fftconvolution(pdf, n: int):
    """Compute the PDF of the average using FFT convolution."""
    l = len(pdf)
    # This is the correct length of the n times convolved pdf
    convolved_size = n * l - n + 1
    # Use it to compute closest optimal length for fast FFT computation
    fft_size = next_fast_len(convolved_size, real=True)

    # Perform FFT on the PDF
    pdf_fft = rfft(pdf, fft_size)

    # n-times convolved F[pdf]
    result_fft = pdf_fft ** n

    # Invert FFT to obtain the (unnormalized) PDF of the sum
    # Resample back to the original array size
    result = irfft(result_fft, l)

    return result
