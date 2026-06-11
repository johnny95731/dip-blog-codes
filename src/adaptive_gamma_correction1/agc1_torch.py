from math import log

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


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to YUV space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB imag in the range of [0, 1] with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`. The range of Y is [0, 1]
        and the range of U and V are [-0.5, 0.5].
    """
    # fmt: off
    matrix = torch.tensor(
        [[ 0.299,  0.587,  0.114],
         [-0.169, -0.331,  0.500],
         [ 0.500, -0.419, -0.081]],
        dtype=__default_dtype(rgb),
        device=rgb.device
    )
    # fmt: on
    yuv = torch.einsum('...oc,...chw->...ohw', matrix, rgb)
    return yuv


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Converts an image from YUV space to RGB space.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be clip to [0, 1].

    Parameters
    ----------
    yuv : torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`.

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape `(*, 3, H, W)`.
    """
    dtype = yuv.dtype if torch.is_floating_point(yuv) else torch.float32
    # fmt: off
    matrix = torch.tensor(
        [[ 1.0, -0.00093, 1.401687],
         [ 1.0, -0.3437, -0.71417],
         [ 1.0,  1.77216, 0.00099]],
        dtype=dtype,
        device=yuv.device
    )
    # fmt: on
    rgb = torch.einsum('...oc,...chw->...ohw', matrix, yuv).clip(0.0, 1.0)
    return rgb


# gagc
def auto_gamma_correction(
    img: torch.Tensor,
    target: float = 0.5,
):
    """Gamma-correction with the automatically estimated gamma.

    1. Computes `gray = rgb_to_gray(rgb)`
    2. Computes mean value of `gray`: `mean(gray)`
    3. Computes `gamma = log(target) / log(mean(gray))`
    4. Applies gamma correction with the computed gamma in step 3.

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    target : float, default=0.5
        Target brightness.

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
    _mean = img.mean((-1, -2, -3), keepdim=True)
    gamma = log(target) / _mean.log()
    res = img.pow(gamma)
    return res


# slagc
def simple_local_gamma_correction(
    rgb: torch.Tensor,
    sigma_blur: float | None = None,
):
    """Adaptive Gamma-correction based on local brightness.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : int, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    num_ch = rgb.size(-3)
    if num_ch == 3:
        gray = rgb.mean(-3, keepdim=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
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
    gamma = log(0.5) / local_mean.add_(1e-8).log_()
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
):
    """Adaptive Gamma-correction based on local brightness.

    1. `gray = rgb_to_gray(rgb)`.
    2. Computes local mean `local_mean`. We use Gaussian lowpass filter in the
       frequency domain to appoximate local mean.
    3. Computes the gamma by `gamma = (local_mean - 0.5) * gain + basic_gamma`.
    4. Gamma correction `res = rgb.pow(gamma)`

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
    sigma_blur : float, default=50
        The sigma for Gaussian blurring. Higher value means the stronger
        blurrness.
    gain : float | None, default=1.3
        The effect of local mean. If both `basic_gamma` and `gain` are None,
        the value will be computed from mean. If only `gain` is None, the
        velue will be 1.3
    gamma_basic : float | None, default=1.0
        The basic gamma value. If `basic_gamma` is `None`, the value will be
        computed from mean.
    gamma_min : float | None, default=None
        The basic gamma value. If `basic_gamma` is `None`, the value will be
        computed from mean.
    gamma_max : float | None, default=None
        The basic gamma value. If `basic_gamma` is `None`, the value will be
        computed from mean.

    Returns
    -------
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    num_ch = rgb.size(-3)
    if num_ch == 3:
        gray = rgb.mean(-3, keepdim=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
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
    # gamma = local_mean * gain + (basic_gamma - 0.5 * gain)
    gamma = local_mean.mul_(gain).add_(gamma_basic - bri_center * gain)

    if gamma_min is None:
        gamma_min = 0
    gamma.clip_(gamma_min, gamma_max)
    res = rgb.pow(gamma)
    return res
