from math import log
from typing import Literal

import cv2
import numpy as np


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
        _sigma = tuple((1 / (2 * np.pi * s)) for s in _sigma)

    freq_y = np.fft.fftfreq(_ksize[0], d=1 / _ksize[0] if d is None else d)
    freq_x = np.fft.rfftfreq(_ksize[1], d=1 / _ksize[1] if d is None else d)
    freq_y = np.divide(np.square(freq_y), -2 * _sigma[0] ** 2).reshape(-1, 1)
    freq_x = np.divide(np.square(freq_x), -2 * _sigma[1] ** 2).reshape(1, -1)
    kernel2d = cv2.exp(freq_y + freq_x)
    if img_f.ndim == 3:
        kernel2d = kernel2d[..., None]
    return kernel2d


# slagc
def simple_local_gamma_correction_scaling(
    rgb: np.ndarray,
    sigma_blur: float | None = None,
    bri_target: float = 0.5,
    brightness: Literal['mean', 'max'] = 'mean',
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image in the range of [0,1] with shape `(H, W, *)`.
    sigma_blur : float | None, default=None
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    bri_target: float, default=0.5
        Balanced brightness value.
    brightness: {"mean", "max"}, default="mean"
        Formula to convert RGB to grayscale.

    Returns
    -------
    np.ndarray
        Enhanced image in the range of [0,1] with the same shape as input.
    """
    assert brightness in ('mean', 'max')
    is_int = np.issubdtype(rgb.dtype, np.integer)
    if is_int:
        max = 255 if np.issubdtype(rgb.dtype, np.uint8) else rgb.max()
        rgb = np.divide(rgb, max, dtype=np.float32)

    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch == 3:
        gray = (
            rgb.max(2, keepdims=True)
            if brightness == 'max'
            else rgb.mean(2, keepdims=True)
        )
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    if sigma_blur is None:
        k = min(gray.shape[:2])
        sigma_blur = 0.3 * (k / 2 - 1) + 0.8
    #
    gray = gray + 1e-8
    gray_f = np.fft.rfft2(gray, axes=(0, 1))
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
    )
    local_mean = gray_f * lowpass
    local_mean = np.fft.irfft2(local_mean, s=gray.shape[:2], axes=(0, 1))

    gamma = log(bri_target) / cv2.log(local_mean + 1e-8, dst=local_mean)
    gamma -= 1
    gray = np.pow(gray, gamma)
    res = rgb * gray
    return res


def simple_local_gamma_correction_yuv(
    rgb: np.ndarray,
    sigma_blur: float | None = None,
    bri_target: float = 0.5,
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image in the range of [0,1] with shape `(H, W, *)`.
    sigma_blur : float | None, default=None
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    bri_target: float, default=0.5
        Balanced brightness value.

    Returns
    -------
    np.ndarray
        Enhanced image in the range of [0,1] with the same shape as input.
    """
    is_int = np.issubdtype(rgb.dtype, np.integer)
    if is_int:
        max = 255 if np.issubdtype(rgb.dtype, np.uint8) else rgb.max()
        rgb = np.divide(rgb, max, dtype=np.float32)

    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch == 3:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        gray = cv2.split(yuv)[0]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    if sigma_blur is None:
        k = min(gray.shape[:2])
        sigma_blur = 0.3 * (k / 2 - 1) + 0.8
    #
    gray = gray + 1e-8
    gray_f = np.fft.rfft2(gray, axes=(0, 1))
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
    )
    local_mean = gray_f * lowpass
    local_mean = np.fft.irfft2(local_mean, s=gray.shape[:2], axes=(0, 1))
    gamma = log(bri_target) / cv2.log(local_mean + 1e-8, dst=local_mean)
    gray = np.pow(gray, gamma)
    if num_ch == 3:
        yuv[..., 0] = gray
        res = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        res = gray
    return res


# lagc
def local_gamma_correction_scaling(
    rgb: np.ndarray,
    sigma_blur: float = 50,
    gain: float | None = 1.3,
    gamma_basic: float | None = 1.0,
    gamma_min: float | None = None,
    gamma_max: float | None = None,
    bri_center: float = 0.5,
    brightness: Literal['mean', 'max'] = 'mean',
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image in the range of [0,1] with shape `(H, W, *)`.
    sigma_blur : float, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    gain : float | None, default=1.3
        The effect of local mean.
    gamma_basic : float | None, default=1.0
        The basic gamma value.
    brightness: {"mean", "max"}, default="mean"
        Formula to convert RGB to grayscale.

    Returns
    -------
    np.ndarray
        Enhanced image in the range of [0,1] with the same shape as input.
    """
    assert brightness in ('mean', 'max')
    is_int = np.issubdtype(rgb.dtype, np.integer)
    if is_int:
        max = 255 if np.issubdtype(rgb.dtype, np.uint8) else rgb.max()
        rgb = np.divide(rgb, max, dtype=np.float32)

    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch > 1:
        gray = (
            rgb.max(2, keepdims=True)
            if brightness == 'max'
            else rgb.mean(2, keepdims=True)
        )
    else:
        gray = rgb
    if gamma_basic is None and gain is None:
        m = gray.mean()
        m_log = np.log(m)
        gamma_basic = log(0.5) / m_log
        gain = -gamma_basic / (2 * m_log * (m + 1e-8))
    elif gamma_basic is None:
        m = gray.mean()
        m_log = np.log(m)
        basic_gamma1 = log(0.5) / m_log
        basic_gamma2 = -gain * (2 * m_log * (m + 1e-8))
        gamma_basic = (basic_gamma1 + basic_gamma2) / 2
    elif gain is None:
        m = gray.mean()
        m_log = np.log(m)
        gain = 2 * m * log(0.5) - 4 * gamma_basic * m * m_log

    gray = gray + 1e-8
    #
    gray_f = np.fft.rfft2(gray, axes=(0, 1))
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
    )
    local_mean = gray_f * lowpass
    local_mean = np.fft.irfft2(local_mean, s=gray.shape[:2], axes=(0, 1))
    # gamma = gain * (local_mean - bri_center) + basic_gamma
    gamma = (local_mean * gain) + (gamma_basic - bri_center * gain)

    if gamma_min is None:
        gamma_min = 0
    np.clip(gamma, gamma_min, gamma_max)
    res = np.pow(rgb, gamma)
    return res


def local_gamma_correction_yuv(
    rgb: np.ndarray,
    sigma_blur: float = 50,
    gain: float | None = 1.3,
    gamma_basic: float | None = 1.0,
    gamma_min: float | None = None,
    gamma_max: float | None = None,
    bri_center: float = 0.5,
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB or grayscale image in the range of [0,1] with shape `(H, W, *)`.
    gain : float | None, default=1.3
        The effect of local mean. If `gain` are `None`, the value will be
        computed from mean.
    gamma_basic : float | None, default=1.0
        The basic gamma value. If `basic_gamma` is `None`, the value will be
        computed from mean.
    gamma_min : float | None, default=None
        The minimum of gamma.
    gamma_max : float | None, default=None
        The maximum of gamma.
    bri_center: float, default=0.5
        Threshold value to determine the pixel is bright or is dark.

    Returns
    -------
    np.ndarray
        Enhanced image in the range of [0,1] with the same shape as input.
    """
    is_int = np.issubdtype(rgb.dtype, np.integer)
    if is_int:
        max = 255 if np.issubdtype(rgb.dtype, np.uint8) else rgb.max()
        rgb = np.divide(rgb, max, dtype=np.float32)

    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if num_ch == 3:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        gray = cv2.split(yuv)[0]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    if gamma_basic is None and gain is None:
        m = gray.mean()
        m_log = np.log(m)
        gamma_basic = log(0.5) / m_log
        gain = -gamma_basic / (2 * m_log * (m + 1e-8))
    elif gamma_basic is None:
        m = gray.mean()
        m_log = np.log(m)
        basic_gamma1 = log(0.5) / m_log
        basic_gamma2 = -gain * (2 * m_log * (m + 1e-8))
        gamma_basic = (basic_gamma1 + basic_gamma2) / 2
    elif gain is None:
        m = gray.mean()
        m_log = np.log(m)
        gain = 2 * m * log(0.5) - 4 * gamma_basic * m * m_log

    gray = gray + 1e-8
    #
    gray_f = np.fft.rfft2(gray, axes=(0, 1))
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
    )
    local_mean = gray_f * lowpass
    local_mean = np.fft.irfft2(local_mean, s=gray.shape[:2], axes=(0, 1))
    # gamma = gain * (local_mean - bri_center) + basic_gamma
    gamma = (local_mean * gain) + (gamma_basic - bri_center * gain)

    if gamma_min is None:
        gamma_min = 0
    np.clip(gamma, gamma_min, gamma_max)
    gray = np.pow(gray, gamma)
    if num_ch == 3:
        yuv[..., 0] = gray
        res = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        res = gray
    return res
