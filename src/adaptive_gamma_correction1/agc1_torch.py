from math import log
from typing import Literal

import torch


# utilities
def __default_dtype(x: torch.Tensor) -> torch.dtype:
    dtype = x.dtype if torch.is_floating_point(x) else torch.float32
    return dtype


def get_gaussian_lowpass(
    img_size: torch.Tensor,
    sigma: float | tuple[float, float],
    d: float | None = 1.0,
    spatial_sigma: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a 2D Gaussian lowpass filter for rfft image.

    Parameters
    ----------
    img_size :torch.Tensor
        The rfft image with shape `(..., H, W)`.
    sigma : float | tuple[float, float]
        The width of Gaussian function. The tuple is `[sigma_y, sigma_x]`.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    spatial_sigma : bool, default=False
        Specify the sigma is for the Gaussian function in spatial domain.
        Set `sigma <- 1 / (2 * pi * sigma)`. Recommend to set `d = 1`.
    dtype : torch.dtype | None, default=None
        The Data type of the filter.
    device : torch.device | None, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Gaussian lowpass filter.

    Notes
    -----
    The relation of Gaussian function in spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / (2 * pi)`

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> lowpass = get_gaussian_lowpass(img_f, 2)
    >>> blurred_f = img_f * lowpass
    >>> blurred = torch.fft.irfft2(blurred_f)
    """
    _ksize = img_size.shape[-2:]
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

    freq_y = (
        torch.fft
        .fftfreq(
            _ksize[0],
            d=1 / _ksize[0] if d is None else d,
            dtype=dtype,
            device=device,
        )
        .square_()
        .div_(-2 * _sigma[0] ** 2)
        .view(-1, 1)
    )
    freq_x = (
        torch.fft
        .rfftfreq(
            _ksize[1],
            d=1 / _ksize[1] if d is None else d,
            dtype=dtype,
            device=device,
        )
        .square_()
        .div_(-2 * _sigma[1] ** 2)
        .view(1, -1)
    )
    kernel2d = (freq_y + freq_x).exp_()
    return kernel2d


# gagc
def auto_gamma_correction(
    img: torch.Tensor,
    target: float = 0.5,
    brightness: Literal['mean', 'max'] = 'mean',
):
    """Gamma-correction with the automatically estimated gamma.

    1. Computes `m = mean(img)`
    2. Computes `gamma = log(target) / log(m)`
    3. Applies gamma correction with the computed `gamma` in step 3.

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    target : float, default=0.5
        Target brightness.
    brightness: {"mean", "max"}, default="mean"
        Formula to convert RGB to grayscale.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.

    References
    ----------
    [1] P. Babakhani1, P. Zarei "Automatic gamma correction based on average
        of brightness," Advances in Computer Science: an International Journal.
        Vol. 4, Issue 6, No.18 , Nov. 2015.
    """
    assert brightness in ('mean', 'max')
    gray = (
        img.amax(-3, keepdim=True)
        if brightness == 'max'
        else img.mean(-3, keepdim=True)
    )
    m = gray.mean((-1, -2), keepdim=True)
    gamma = log(target) / m.log()
    res = img.pow(gamma)
    return res


# slagc
def simple_local_gamma_correction(
    rgb: torch.Tensor,
    sigma_blur: float | None = None,
    bri_target: float = 0.5,
    brightness: Literal['mean', 'max'] = 'mean',
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : int, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    bri_target: float, default=0.5
        Balanced brightness value.
    brightness: {"mean", "max"}, default="mean"
        Formula to convert RGB to grayscale.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    assert brightness in ('mean', 'max')
    num_ch = rgb.size(-3)
    if num_ch > 1:
        gray = (
            rgb.amax(-3, keepdim=True)
            if brightness == 'max'
            else rgb.mean(-3, keepdim=True)
        )
    else:
        gray = rgb
    if sigma_blur is None:
        k = min(gray.shape[-2:])
        sigma_blur = 0.3 * (k / 2 - 1) + 0.8

    dtype = __default_dtype(gray)
    gray_f = torch.fft.rfft2(gray)
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
        dtype=dtype,
        device=gray.device,
    )
    local_mean = gray_f.mul_(lowpass)
    local_mean = torch.fft.irfft2(local_mean, s=gray.shape[-2:])
    #
    gamma = log(bri_target) / local_mean.add_(1e-8).log_()
    res = rgb.pow(gamma)
    return res


def local_gamma_correction(
    rgb: torch.Tensor,
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
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : float, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
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
    brightness: {"mean", "max"}, default="mean"
        Formula to convert RGB to grayscale.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    assert brightness in ('mean', 'max')
    num_ch = rgb.size(-3)
    if num_ch > 1:
        gray = (
            rgb.amax(-3, keepdim=True)
            if brightness == 'max'
            else rgb.mean(-3, keepdim=True)
        )
    else:
        gray = rgb
    if gamma_basic is None and gain is None:
        m = gray.mean((-1, -2), keepdim=True)
        m_log = m.log()
        gamma_basic = log(0.5) / m_log
        gain = -gamma_basic / (2 * m_log * (m + 1e-8))
    elif gamma_basic is None:
        m = gray.mean((-1, -2), keepdim=True)
        m_log = m.log()
        basic_gamma1 = log(0.5) / m_log
        basic_gamma2 = -gain * (2 * m_log * (m + 1e-8))
        gamma_basic = (basic_gamma1 + basic_gamma2) / 2
    elif gain is None:
        m = gray.mean((-1, -2), keepdim=True)
        m_log = m.log()
        gain = 2 * m * log(0.5) - 4 * gamma_basic * m * m_log

    gray = gray.add(1e-8)
    #
    dtype = __default_dtype(gray)
    gray_f = torch.fft.rfft2(gray)
    lowpass = get_gaussian_lowpass(
        gray_f,
        sigma_blur,
        d=1.0,
        spatial_sigma=True,
        dtype=dtype,
        device=gray.device,
    )
    local_mean = gray_f.mul_(lowpass)
    local_mean = torch.fft.irfft2(local_mean, s=gray.shape[-2:])
    # gamma = gain * (local_mean - bri_center) + basic_gamma
    gamma = local_mean.mul_(gain).add_(gamma_basic - bri_center * gain)

    if gamma_min is None:
        gamma_min = 0
    gamma.clip_(gamma_min, gamma_max)
    res = rgb.pow(gamma)
    return res
