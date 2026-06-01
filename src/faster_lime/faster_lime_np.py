from typing import Literal

import cv2
import torch
import numpy as np


# Utilities
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


def get_gaussian_highpass(
    img_f: int | tuple[int, int] | np.ndarray,
    sigma: float | tuple[float, float],
    d: float | None = 1.0,
    spatial_sigma: bool = False,
) -> np.ndarray:
    """Create a 2D Gaussian highpass filter for rfft image.

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
        Set `sigma <- 1 / (2 * pi * sigma)`.

    Returns
    -------
    np.ndarray
        2D Gaussian highpass filter.

    Notes
    -----
    The relation of Gaussian function in spatial domain and frequency domain
    is `sigma_s * sigma_f = 1 / (2 * pi)`

    Examples
    --------

    >>> img_f = np.fft.rfft2(img)
    >>> highpass = get_gaussian_highpass(img_f, 2)
    >>> edge_f = img_f * highpass
    >>> edge = np.fft.irfft2(edge_f)
    """
    kernel = get_gaussian_lowpass(img_f, sigma, d, spatial_sigma)
    kernel = np.subtract(1.0, kernel, out=kernel)  # highpass
    return kernel


def get_butterworth_lowpass(
    img_f: np.ndarray,
    cutoff: float,
    order: float = 1.0,
    d: float | None = 1.0,
) -> np.ndarray:
    """Create a 2D Butterworth lowpass filter for rfft image.

    Parameters
    ----------
    img_f : np.ndarray
        The rfft image with shape `(H, W)`.
    cutoff : float
        The cutoff frequency of Butterworth filter.
    order : float, default=1.0
        The order of Butterworth filter.
    d : float | None, default=1.0
        The sampling length scale. If `None`, uses `1/img_size`. For details,
        see `np.fft.fftfreq` and `np.fft.rfftfreq`.

    Returns
    -------
    np.ndarray
        2D Butterworth lowpass filter.

    Examples
    --------

    >>> img_f = np.fft.rfft2(img)
    >>> lowpass = get_butterworth_lowpass(img_f, 10)
    >>> blurred_f = img_f * lowpass
    >>> blurred = np.fft.irfft2(blurred_f)
    """
    _ksize = img_f.shape[:2]
    _ksize = _ksize[0], 2 * _ksize[1] - 2
    if not isinstance(cutoff, (int, float)):
        raise TypeError(f'Invalid type of `cutoff`: {type(cutoff)}')
    elif cutoff <= 0:
        raise ValueError(f'`cutoff` must be positive: {cutoff}')
    if not isinstance(order, (int, float)):
        raise TypeError(f'Invalid type of `order`: {type(order)}')
    elif order <= 0:
        raise ValueError(f'`order` must be positive: {order}')

    freq_y = np.fft.fftfreq(
        _ksize[0], d=1 / _ksize[0] if d is None else d
    ).reshape(-1, 1)
    freq_x = np.fft.rfftfreq(
        _ksize[1], d=1 / _ksize[1] if d is None else d
    ).reshape(1, -1)
    kernel2d = np.square(freq_y) + np.square(freq_x)  # D(u, v)**2
    # 1 / [1 + (D(u, v) / cutoff) ** (2 * order)]
    kernel2d = np.divide(kernel2d, cutoff**2, out=kernel2d)
    kernel2d = cv2.pow(kernel2d, order, dst=kernel2d) + 1.0
    kernel2d = np.reciprocal(kernel2d)
    if img_f.ndim == 3:
        kernel2d = kernel2d[..., None]
    return kernel2d


def get_freq_laplacian(
    img_f: int | tuple[int, int] | np.ndarray,
    form: Literal['continuous', '5-point', '9-point'] = 'continuous',
    d: float | None = 1.0,
) -> np.ndarray:
    """Create a frequency domain Laplacian filter for rfft image.

    Parameters
    ----------
    img_f : np.ndarray
        The rfft image with shape `(H, W, *)`.
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
        see `np.fft.fftfreq` and `np.fft.rfftfreq`.

    Returns
    -------
    np.ndarray
        2D Laplacian filter in frequency domain.

    Notes
    -----
    The results of 4-neighbor and 8-neighbor (in frequency domain) are similar
    to the result by using the convolution. Theese options are present for
    solving PDE in the frequency domain.

    Examples
    --------

    >>> img_f = np.fft.rfft2(img)
    >>> highpass = get_freq_laplacian(img_f, 10)
    >>> edge_f = img_f * highpass
    >>> edge = np.fft.irfft2(edge_f)
    """
    _ksize = img_f.shape[:2]
    _ksize = _ksize[0], 2 * _ksize[1] - 2

    freq_y = np.fft.fftfreq(
        _ksize[0], d=1 / _ksize[0] if d is None else d
    ).reshape(-1, 1)  # type: np.ndarray
    freq_x = np.fft.rfftfreq(
        _ksize[1], d=1 / _ksize[1] if d is None else d
    ).reshape(1, -1)  # type: np.ndarray

    if form == 'continuous':
        # Discretize the Fourier transform of the continuous Laplacian operator
        freq_y = np.square(2 * np.pi * freq_y)
        freq_x = np.square(2 * np.pi * freq_x)
        fft_laplacian = -(freq_y + freq_x)
    elif form == '5-point':
        # The Fourier transform of the 5-point stencil.
        freq_y = np.cos(2 * np.pi * freq_y) * 2.0 - 4.0
        freq_x = np.cos(2 * np.pi * freq_x) * 2.0
        fft_laplacian = freq_y + freq_x
    elif form == '9-point':
        # The Fourier transform of the 9-point stencil
        freq_y = np.cos(2 * np.pi * freq_y) * 2.0
        freq_x = np.cos(2 * np.pi * freq_x) * 2.0
        fft_laplacian = (freq_y + freq_x) + (freq_y * freq_x) - 8.0
    else:
        raise ValueError(
            f'`form` must be one of "continuous", "5-point", or "9-point": {form}'
        )
    if img_f.ndim == 3:
        fft_laplacian = fft_laplacian[..., None]
    return fft_laplacian


# Main functions
def lime_screened_poisson(
    img: np.ndarray,
    alpha: float = 2.0,
    gamma: float = 0.5,
):
    """A faster LIME by screened Poisson equation. Original LIME algorithm [1].

    Parameters
    ----------
    img : np.ndarray
        Image in the range of [0, 1] with shape `(H, W, *)`.
    alpha : float, default=2
        Blurrness, by default 2
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing,
        by default 0.5.

    Returns
    -------
    np.ndarray
        Enhanced image. shape `(H, W, *)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    num_ch = 1 if img.ndim == 2 else img.shape[2]
    if num_ch == 3:
        t_hat = img.max(2, keepdims=True)
    elif num_ch == 1:
        t_hat = img
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    t_hat_f = np.fft.rfft2(t_hat, axes=(0, 1))
    fft_laplacian = get_freq_laplacian(t_hat_f, form='continuous', d=1)
    kernelfft = cv2.addWeighted(
        fft_laplacian, -alpha, 0, 0, 1.0, dst=fft_laplacian
    )
    res_f = np.divide(t_hat_f, kernelfft)
    img_t = np.fft.irfft2(res_f, s=t_hat.shape[:2], axes=(0, 1))

    img_t.clip(0.0, None, out=img_t)
    cv2.pow(img_t, gamma, dst=img_t)
    res = img / (img_t + 1e-8)
    np.clip(res, 0.0, 1.0, out=res)
    return res


def lime_butterworth(
    img: np.ndarray,
    alpha: float = 2.0,
    order: float = 1.0,
    gamma: float = 0.5,
):
    """A faster LIME by Butterworth lowpass filter. Original LIME algorithm [1].

    Parameters
    ----------
    img : np.ndarray
        Image in the range of [0, 1] with shape `(H, W, C)`.
    alpha : float, default=2
        Blurrness.
    order : float, default=1
        The order of Butterworth filter.
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing.

    Returns
    -------
    np.ndarray
        Enhanced image. shape `(H, W, C)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    num_ch = 1 if img.ndim == 2 else img.shape[2]
    if num_ch == 3:
        t_hat = img.max(2, keepdims=True)
    elif num_ch == 1:
        t_hat = img
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    t_hat_f = np.fft.rfft2(t_hat, axes=(0, 1))
    fft_filter = get_butterworth_lowpass(t_hat_f, 1 / alpha, order, d=1)
    res_f = np.divide(t_hat_f, fft_filter, out=t_hat_f)
    img_t = np.fft.irfft2(res_f, s=t_hat.shape[:2], axes=(0, 1))
    #
    mini = img_t.min(keepdims=True)
    maxi = img_t.max(keepdims=True)
    img_t = np.divide(img_t - mini, maxi - mini, out=img_t)

    img_t.clip(0.0, None, out=img_t)
    cv2.pow(img_t, gamma, dst=img_t)
    res = img / (img_t + 1e-6)
    np.clip(res, 0.0, 1.0, out=res)
    return res


def lime_blurred_screened(
    img: np.ndarray,
    alpha: float = 2.0,
    sigma: float = 10.0,
    gamma: float = 0.5,
):
    """A faster LIME by blurred the gradient in L2 norm. Original
    LIME algorithm [1].
    Parameters
    ----------
    img : np.ndarray
        Image in the range of [0, 1] with shape `(H, W, C)`.
    alpha : float, default=2
        Blurrness.
    sigma : float, default=10
        An argument for Gaussian filter.
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing.

    Returns
    -------
    np.ndarray
        Enhanced image. shape `(H, W, C)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    num_ch = 1 if img.ndim == 2 else img.shape[2]
    if num_ch == 3:
        t_hat = img.max(2, keepdims=True)
    elif num_ch == 1:
        t_hat = img
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    t_hat_f = np.fft.rfft2(t_hat, axes=(0, 1))
    fft_filter = get_gaussian_lowpass(t_hat_f, sigma, d=1, spatial_sigma=True)
    fft_laplacian = get_freq_laplacian(t_hat_f, form='continuous', d=1)
    fft_filter = -alpha * fft_laplacian * fft_filter + 1.0
    res_f = np.divide(t_hat_f, fft_filter, out=t_hat_f)
    img_t = np.fft.irfft2(res_f, s=t_hat.shape[:2], axes=(0, 1))

    img_t.clip(0.0, None, out=img_t)
    cv2.pow(img_t, gamma, dst=img_t)
    res = img / (img_t + 1e-8)
    np.clip(res, 0.0, 1.0, out=res)
    return res


def lime_gaussian_highpass(
    img: np.ndarray,
    alpha: float = 2.0,
    sigma: float = 5.0,
    gamma: float = 0.5,
):
    """A faster LIME by replacing laplacian by Gaussian highpass filter.
    Original LIME algorithm [1].

    Parameters
    ----------
    img : np.ndarray
        Image in the range of [0, 1] with shape `(H, W, C)`.
    alpha : float, default=2
        Blurrness.
    sigma : float, default=5
        An argument for Gaussian filter
    gamma : float, default=0.5
        Gamma correction the esimated illuminant befoe enhancing,

    Returns
    -------
    np.ndarray
        Enhanced image. shape `(H, W, C)`.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    num_ch = 1 if img.ndim == 2 else img.shape[2]
    if num_ch == 3:
        t_hat = img.max(2, keepdims=True)
    elif num_ch == 1:
        t_hat = img
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    t_hat_f = np.fft.rfft2(t_hat, axes=(0, 1))
    fft_filter = get_gaussian_highpass(t_hat_f, 1 / sigma)
    t_hat_f /= alpha * fft_filter + 1.0
    img_t = np.fft.irfft2(t_hat_f, s=t_hat.shape[:2], axes=(0, 1))

    img_t.clip(0.0, None, out=img_t)
    cv2.pow(img_t, gamma, dst=img_t)
    res = img / (img_t + 1e-6)
    np.clip(res, 0.0, 1.0, out=res)
    return res
