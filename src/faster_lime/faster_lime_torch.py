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


def get_gaussian_highpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    sigma: float | tuple[float, float],
    d: float | None = 1.0,
    spatial_sigma: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a 2D Gaussian highpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. The tuple is `(size_y, size_x)`. Or,
        the rfft image with shape `(..., H, W)`.
    sigma : float | tuple[float, float]
        The width of Gaussian function. The tuple is `[sigma_y, sigma_x]`.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    spatial_sigma : bool, default=False
        Specify the sigma is for the Gaussian function in spatial domain.
        Set `sigma <- 1 / (2 * pi * sigma)`.
    dtype : torch.dtype | None, default=None
        The Data type of the filter.
    device : torch.device | None, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Gaussian highpass filter.

    Notes
    -----
    The relation of Gaussian function in spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / (2 * pi)`

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> highpass = get_gaussian_highpass(img_f, 2)
    >>> edge_f = img_f * highpass
    >>> edge = torch.fft.irfft2(edge_f)
    """
    kernel = get_gaussian_lowpass(
        img_size, sigma, d, spatial_sigma, dtype, device
    )
    kernel = torch.sub(1.0, kernel, out=kernel)  # highpass
    return kernel


def get_butterworth_lowpass(
    img_size: int | tuple[int, int] | torch.Tensor,
    cutoff: float,
    order: float = 1.0,
    d: float | None = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a 2D Butterworth lowpass filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. The tuple is `(size_y, size_x)`. Or,
        the rfft image with shape `(..., H, W)`.
    cutoff : float
        The cutoff frequency of Butterworth filter.
    order : float, default=1.0
        The order of Butterworth filter.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype | None, default=None
        The Data type of the filter.
    device : torch.device | None, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Butterworth lowpass filter.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> lowpass = get_butterworth_lowpass(img_f, 10)
    >>> blurred_f = img_f * lowpass
    >>> blurred = torch.fft.irfft2(blurred_f)
    """
    _ksize = img_size.shape[-2:]
    _ksize = _ksize[0], 2 * _ksize[1] - 2
    if not isinstance(cutoff, (int, float)):
        raise TypeError(f'Invalid type of `cutoff`: {type(cutoff)}')
    elif cutoff <= 0:
        raise ValueError(f'`cutoff` must be positive: {cutoff}')
    if not isinstance(order, (int, float)):
        raise TypeError(f'Invalid type of `order`: {type(order)}')
    elif order <= 0:
        raise ValueError(f'`order` must be positive: {order}')

    freq_y = torch.fft.fftfreq(
        _ksize[0],
        d=1 / _ksize[0] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(-1, 1)
    freq_x = torch.fft.rfftfreq(
        _ksize[1],
        d=1 / _ksize[1] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(1, -1)
    kernel2d = freq_y.square_() + freq_x.square_()  # D(u, v)**2
    # 1 / [1 + (D(u, v) / cutoff) ** (2 * order)]
    kernel2d.div_(cutoff**2).pow_(order).add_(1.0).reciprocal_()
    return kernel2d


def get_freq_laplacian(
    img_size: int | tuple[int, int] | torch.Tensor,
    form: Literal['continuous', '5-point', '9-point'] = 'continuous',
    d: float | None = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a frequency domain Laplacian filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. The tuple is `(size_y, size_x)`. Or,
        the rfft image with shape `(..., H, W)`.
    form : {'continuous', '5-point', '9-point'}, default='continuous'
        The form of approximation of discrete Laplacian filter in frequency
        domain.

        - `'continuous'`: Discretize the Fourier transform of the continuous
        Laplacian operator. Better
        - `'5-point'`: Computes the Fourier transform of the 5-point stencil
        Laplacian operator.
        - `'9-point'`: Computes the Fourier transform of the 9-point stencil
        Laplacian operator.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype | None, default=None
        The Data type of the filter.
    device : torch.device | None, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Laplacian filter in frequency domain.

    Notes
    -----
    The results of 4-neighbor and 8-neighbor (in frequency domain) are similar
    to the result by using the convolution. Theese options are present for
    solving PDE in the frequency domain.

    Examples
    --------

    >>> img_f = torch.fft.rfft2(img)
    >>> highpass = get_freq_laplacian(img_f, 10)
    >>> edge_f = img_f * highpass
    >>> edge = torch.fft.irfft2(edge_f)
    """
    _ksize = img_size.shape[-2:]
    _ksize = _ksize[0], 2 * _ksize[1] - 2

    freq_y = torch.fft.fftfreq(
        _ksize[0],
        d=1 / _ksize[0] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(-1, 1)  # type: torch.Tensor
    freq_x = torch.fft.rfftfreq(
        _ksize[1],
        d=1 / _ksize[1] if d is None else d,
        dtype=dtype,
        device=device,
    ).view(1, -1)  # type: torch.Tensor

    if form == 'continuous':
        # Discretize the Fourier transform of the continuous Laplacian operator
        freq_y.mul_(2 * torch.pi).square_()
        freq_x.mul_(2 * torch.pi).square_()
        fft_laplacian = freq_y.add(freq_x).neg_()
    elif form == '5-point':
        # The Fourier transform of the 5-point stencil.
        freq_y.mul_(2 * torch.pi).cos_().mul_(2.0).sub_(4.0)
        freq_x.mul_(2 * torch.pi).cos_().mul_(2.0)
        fft_laplacian = freq_y + freq_x
    elif form == '9-point':
        # The Fourier transform of the 9-point stencil
        freq_y.mul_(2 * torch.pi).cos_().mul_(2.0)
        freq_x.mul_(2 * torch.pi).cos_().mul_(2.0)
        fft_laplacian = (freq_y + freq_x).add_(freq_y * freq_x).sub_(8.0)
    else:
        raise ValueError(
            f'`form` must be one of "continuous", "5-point", or "9-point": {form}'
        )
    return fft_laplacian


# main functions
def lime_screened_poisson(
    img: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 0.5,
):
    """A faster LIME by screened Poisson equation. Original LIME algorithm [1].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape `(*, C, H, W)`.
    alpha : float, default=2
        Blurrness, by default 2
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing

    Returns
    -------
    torch.Tensor
        Enhanced image. shape `(*, C, H, W)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = __default_dtype(img)
    device = img.device

    t_hat_f = torch.fft.rfft2(t_hat)
    fft_laplacian = get_freq_laplacian(
        t_hat_f, form='continuous', d=1, dtype=dtype, device=device
    )
    res_f = t_hat_f.div_(fft_laplacian.mul_(-alpha).add_(1.0))
    img_t = torch.fft.irfft2(res_f, s=t_hat.shape[-2:], out=t_hat)

    img_t.relu_()
    img_t **= gamma
    res = img.div(img_t.add_(1e-8)).clip_(0.0, 1.0)
    return res


def lime_butterworth(
    img: torch.Tensor,
    alpha: float = 2.0,
    order: float = 1.0,
    gamma: float = 0.5,
):
    """A faster LIME by Butterworth lowpass filter. Original LIME algorithm [1].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape `(*, C, H, W)`.
    alpha : float, default=2
        Blurrness
    order : float, default=1
        The order of Butterworth filter
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing

    Returns
    -------
    torch.Tensor
        Enhanced image. shape `(*, C, H, W)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = __default_dtype(img)
    device = img.device

    t_hat_f = torch.fft.rfft2(t_hat)
    fft_filter = get_butterworth_lowpass(
        t_hat_f, 1 / alpha, order, d=1, dtype=dtype, device=device
    )
    res_f = t_hat_f.mul_(fft_filter)
    img_t = torch.fft.irfft2(res_f, s=t_hat.shape[-2:], out=t_hat)
    mini = img_t.amin((-1, -2), keepdim=True)
    maxi = img_t.amax((-1, -2), keepdim=True)
    img_t.sub_(mini).div_(maxi.sub_(mini))

    img_t.relu_()
    img_t **= gamma
    res = img.div(img_t.add_(1e-8)).clip_(0.0, 1.0)
    return res


def lime_blurred_screened(
    img: torch.Tensor,
    alpha: float = 2.0,
    sigma: float = 10.0,
    gamma: float = 0.5,
):
    """A faster LIME by blurred the gradient in L2 norm. Original
    LIME algorithm [1].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape `(*, C, H, W)`.
    alpha : float, default=2
        Blurrness
    sigma : float, default=10
        An argument for Gaussian filter.
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing.

    Returns
    -------
    torch.Tensor
        Enhanced image. shape `(*, C, H, W)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = __default_dtype(img)
    device = img.device

    t_hat_f = torch.fft.rfft2(t_hat)
    fft_filter = get_gaussian_lowpass(
        t_hat_f, 1 / (2 * torch.pi * sigma), d=1, dtype=dtype, device=device
    )
    fft_laplacian = get_freq_laplacian(
        t_hat_f, form='continuous', d=1, dtype=dtype, device=device
    )
    res_f = t_hat_f.div_(fft_laplacian.mul_(-alpha).mul_(fft_filter).add_(1.0))
    img_t = torch.fft.irfft2(res_f, s=t_hat.shape[-2:], out=t_hat)

    img_t.relu_()
    img_t **= gamma
    res = img.div(img_t.add_(1e-8)).clip_(0.0, 1.0)
    return res


def lime_gaussian_highpass(
    img: torch.Tensor,
    alpha: float = 2.0,
    sigma: float = 5.0,
    gamma: float = 0.5,
):
    """A faster LIME by replacing laplacian by Gaussian highpass filter.
    Original LIME algorithm [1].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape `(*, C, H, W)`.
    alpha : float, default=2
        Blurrness
    sigma : float, optional
        An argument for Gaussian filter, by default 5
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing,

    Returns
    -------
    torch.Tensor
        Enhanced image. shape `(*, C, H, W)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = __default_dtype(img)
    device = img.device

    t_hat_f = torch.fft.rfft2(t_hat)
    fft_filter = get_gaussian_highpass(
        t_hat_f, 1 / sigma, d=1, dtype=dtype, device=device
    )
    res_f = t_hat_f.div_(fft_filter.mul_(alpha).add_(1.0))
    img_t = torch.fft.irfft2(res_f, s=t_hat.shape[-2:], out=t_hat)

    img_t.relu_()
    img_t **= gamma
    res = img.div(img_t.add_(1e-8)).clip_(0.0, 1.0)
    return res
