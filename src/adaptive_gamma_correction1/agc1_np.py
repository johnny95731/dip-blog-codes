from math import log

import cv2
import numpy as np
import torch


# utilities
def get_gaussian_lowpass(
    img_f: np.ndarray,
    sigma: float | tuple[float, float],
    d: float | None = 1.0,
    spatial_sigma: bool = False,
) -> np.ndarray:
    """Create a 2D Gaussian lowpass filter for rfft image.

    Parameters
    ----------
    img_f : np.ndarray
        The rfft image with shape `(H, W, *)`.
    sigma : float | tuple[float, float]
        The width of Gaussian function. The tuple is `[sigma_y, sigma_x]`.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `np.fft.fftfreq` and `np.fft.rfftfreq`.
    spatial_sigma : bool, default=False
        Specify the sigma is for the Gaussian function in spatial domain.
        Set `sigma <- 1 / (2 * pi * sigma)`. Recommend to set `d = 1`.

    Returns
    -------
    np.ndarray
        2D Gaussian lowpass filter.

    Notes
    -----
    The relation of Gaussian function in spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / (2 * pi)`

    Examples
    --------

    >>> img_f = np.fft.rfft2(img)
    >>> lowpass = get_gaussian_lowpass(img_f, 2)
    >>> blurred_f = img_f * lowpass
    >>> blurred = np.fft.irfft2(blurred_f)
    """
    _ksize = img_f.shape[:2]
    _ksize = _ksize[0], 2 * _ksize[1] - 2
    if isinstance(sigma, (int, float)):
        _sigma = (sigma, sigma)
    elif isinstance(sigma, (tuple, list)):
        if len(sigma) == 0:
            raise ValueError('len(gamma) can not be 0.')
        elif len(sigma) == 1:
            _sigma = (sigma[0], sigma[0])
        else:
            _sigma = (sigma[0], sigma[1])
    else:
        raise TypeError(f'Invalid type of `gamma`: {type(sigma)}')
    if spatial_sigma:
        _sigma = tuple((1 / (2 * torch.pi * s)) for s in _sigma)

    freq_y = np.fft.fftfreq(_ksize[0], d=1 / _ksize[0] if d is None else d)
    freq_x = np.fft.rfftfreq(_ksize[1], d=1 / _ksize[1] if d is None else d)
    freq_y = np.divide(np.square(freq_y), -2 * _sigma[0] ** 2).reshape(-1, 1)
    freq_x = np.divide(np.square(freq_x), -2 * _sigma[1] ** 2).reshape(1, -1)
    kernel2d = cv2.exp(freq_y + freq_x)
    if img_f.ndim == 3:
        kernel2d = kernel2d[..., None]
    return kernel2d


# gagc
def auto_gamma_correction(
    img: np.ndarray,
    target: float = 0.5,
):
    """Gamma-correction with the automatically estimated gamma.

    1. Computes `gray = rgb_to_gray(rgb)`
    2. Computes mean value of `gray`: `mean(gray)`
    3. Computes `gamma = log(target) / log(mean(gray))`
    4. Applies gamma correction with the computed gamma in step 3.

    Parameters
    ----------
    img : np.ndarray
        An RGB or grayscale image with shape `(H, W, *)`.
    target : float, default=0.5
        Target brightness.

    Returns
    -------
    np.ndarray
        Enhanced image with the same shape as input.

    References
    ----------
    [1] P. Babakhani1, P. Zarei "Automatic gamma correction based on average
        of brightness," Advances in Computer Science: an International Journal.
        Vol. 4, Issue 6, No.18 , Nov. 2015.
    """
    _mean = img.mean()
    gamma = (log(target) / np.log(_mean)).item()
    res = cv2.pow(img, gamma)
    return res


# slagc
def simple_local_gamma_correction(
    rgb: np.ndarray,
    sigma_blur: float = 10,
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image with shape `(H, W, *)`.
    sigma_blur : int, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.

    Returns
    -------
    np.ndarray
        Enhanced image with the same shape as input.
    """
    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch == 3:
        gray = rgb.mean(2, keepdims=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    #
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigma_blur, sigma_blur)
    gamma = log(0.5) / cv2.log(local_mean + 1e-8, dst=local_mean)
    if num_ch == 3:
        gamma = gamma[..., None]
    res = np.pow(rgb, gamma)
    return res


def local_gamma_correction(
    rgb: np.ndarray,
    sigma_blur: float = 50,
    basic_gamma: float = 1.0,
    gain: float = 1.3,
):
    """Adaptive Gamma-correction based on local brightness.

    1. `gray = rgb_to_gray(rgb)`.
    2. Computes local mean `local_mean`. We use Gaussian lowpass filter in the
       frequency domain to appoximate local mean.
    3. Computes the gamma by `gamma = (local_mean - 0.5) * gain + basic_gamma`.
    4. Gamma correction `res = rgb.pow(gamma)`

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : float, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    basic_gamma : float, default=1.0
        The basic gamma value.
    gain : float, default=1.3
        The effect of local mean.

    Returns
    -------
    np.ndarray
        Enhanced image with the same shape as input.
    """
    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch == 3:
        gray = rgb.mean(2, keepdims=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    gray = gray + 1e-8
    #
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigma_blur, sigma_blur)
    # gamma = local_mean * gain + (basic_gamma - 0.5 * gain)
    gamma = (local_mean * gain) + (basic_gamma - 0.5 * gain)
    np.clip(gamma, 0.0, None)
    if num_ch == 3:
        gamma = gamma[..., None]
    res = np.pow(rgb, gamma)
    return res
