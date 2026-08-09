from typing import Literal

import cv2
import numpy as np
from scipy import special


# Utils
def _gaussian_cdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    # Z-score
    z_score = x - mean
    np.multiply(std, 2**0.5, out=std)
    np.divide(z_score, std, out=z_score)
    # cdf
    res = special.erf(z_score, out=z_score)
    cv2.addWeighted(res, 0.5, 0, 0, 0.5, dst=res)
    return res


def _laplace_cdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    diff = x - mean
    sign = np.sign(diff)
    z_score = np.divide(np.abs(diff, out=diff), std, out=diff)
    np.multiply(diff, -(2**0.5), out=diff)
    # cdf
    res = cv2.exp(z_score, dst=z_score)
    np.multiply(res, sign, out=res)
    np.subtract(sign, res, out=res)
    cv2.addWeighted(res, 0.5, 0, 0, 0.5, dst=res)
    return res


def _cauchy_cdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z_score = x - mean
    np.divide(z_score, std, out=z_score)

    res = np.arctan(z_score, out=z_score)
    cv2.addWeighted(res, 1 / np.pi, 0, 0, 0.5, dst=res)
    return res


def _logistic_cdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z_score = x - mean
    np.divide(z_score, std, out=z_score)

    res = special.expit(z_score, out=z_score)
    return res


def _hyperbolic_cdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z_score = x - mean
    np.divide(z_score, std, out=z_score)

    res = np.multiply(z_score, np.pi / 2, out=z_score)
    cv2.exp(res, dst=res)
    np.arctan(res, out=res)
    np.multiply(res, 2 / np.pi, out=res)
    return res


def convex_comb(x: np.ndarray, y: np.ndarray, c: float):
    """res = c * x + (1-c) * y"""
    if abs(c) < 1e-10:
        res = y
    elif abs(1 - c) < 1e-10:
        res = x
    else:
        res = cv2.addWeighted(x, c, y, 1 - c, 0)
    return res


#
def modified_lide(
    rgb: np.ndarray,
    radius: int | None = 50,
    eta_m: float = 0.6,
    eta_s: float = 0.7,
    std_min: float = 0.2,
    std_max: float | None = None,
    alpha: float = 0,
    gamma: float = 1,
    distrib: Literal[
        'cauchy', 'gaussian', 'hyperbolic', 'laplace', 'logistic'
    ] = 'gaussian',
):
    """Automatic contrast enhancement by modifying local intensity
    distribution equalization (LIDE) [1].

    1. Convert image `I` to grayscale `g_in`.
    2. Computes local stats and global stats (mean and std).
    3. Convex combination `mean = eta_m * m_global + (1-eta_m) * m_local` and
       `std = eta_s * s_global + (1-eta_s) * s_local`.
    4. Translate mean `center = mean - alpha * std`.
    5. Clip standard deviation `std = clip(std, std_min, std_max)`.
    6. Intensity transform by the CDF of probability distribution:
       `g_out = F(g_in, center, std)`.
    7. Tone mapping `O = (g_out/g_in) ** gamma * I`.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image with shape `(H, W, C)` or `(H, W)`.
    radius : int | None, default=51
        The radius of kernel of the mean filter. `None` means computing global
        stats instead of local stats.
    eta_m : float, default=0.6
        The convex combination factor for mean value.
        `mean = eta_m * m_global + (1-eta_m) * m_local`
    eta_s : float, default=0.7
        The convex combination factor for standard deviation.
        `std = eta_s * s_global + (1-eta_s) * s_local`
    std_min : float, default=0.02
        The minimum value of the standard deviation.
    std_max : float | None, default=None
        The maximum value of the standard deviation.
    alpha : float, alpha = 0
        Brightness controls. A larger value means brighter result.
    gamma : float, alpha = 1
        Stength of the enhancement. A larger value means stronger contrast.
    distrib : {"cauchy", "gaussian", "hyperbolic", "laplace", "logistic"}, default = "gaussian"
        The distribution model.

    Returns
    -------
    np.ndarray
        Enhanced image with the same shape as input.

    References
    ------
    [1] Marukatat, S. Image enhancement using local intensity distribution
        equalization. J Image Video Proc. 2015, 31 (2015).
        https://doi.org/10.1186/s13640-015-0085-2
    """
    assert 3 <= rgb.ndim <= 4
    assert distrib in (
        valid_li := (
            'cauchy',
            'gaussian',
            'hyperbolic',
            'laplace',
            'logistic',
        )
    ), f'Invalid value of `model`: {distrib}, valid values: {valid_li}'
    assert radius is None or (isinstance(radius, int) and radius >= 1), (
        f'`ksize` must be `None` or an positive integer: {radius}'
    )
    assert isinstance(std_min, (int, float)) and std_min > 0, (
        f'`std_min` must be positive: {std_min}'
    )
    shape = rgb.shape
    if len(shape) == 3 and shape[2] == 3:
        gray = np.mean(rgb, 2)
        is_rgb = True
    elif len(shape) == 1:
        gray = rgb
        is_rgb = False
    else:
        raise ValueError(f'Invalid shape of image: {shape}')
    gray = gray + 1e-8
    #

    mean, std = cv2.meanStdDev(gray)
    if radius is not None:
        ksize = 2 * radius + 1
        local_mean = cv2.boxFilter(gray, -1, (ksize, ksize))
        sq_mean = cv2.boxFilter(gray * gray, -1, (ksize, ksize))
        # std(x) = mean(x**2) - mean(x)**2
        local_std = np.subtract(sq_mean, mean * mean, out=sq_mean)

        mean = convex_comb(mean, local_mean, eta_m)
        std = convex_comb(std, local_std, eta_s)
    #
    std = np.clip(std, std_min, std_max, out=std)
    center = (mean - alpha * std) if abs(alpha) > 1e-10 else mean

    if distrib == 'cauchy':
        res = _cauchy_cdf(gray, center, std)
    elif distrib == 'gaussian':
        res = _gaussian_cdf(gray, center, std)
    elif distrib == 'hyperbolic':
        res = _hyperbolic_cdf(gray, center, std)
    elif distrib == 'laplace':
        res = _laplace_cdf(gray, center, std)
    elif distrib == 'logistic':
        res = _logistic_cdf(gray, center, std)

    use_gamma = abs(gamma - 1) > 1e-10
    if is_rgb and use_gamma:
        res = rgb * cv2.pow((res / gray)[..., None], gamma)
    elif is_rgb:
        res = rgb * (res / gray)[..., None]
    elif use_gamma:
        res = cv2.pow(res, gamma)
    np.clip(res, 0.0, 1.0, out=res)
    return res
