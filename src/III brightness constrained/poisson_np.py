from typing import Literal

import numpy as np


# utilities
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
def bri_constrained_poisson(
    img: np.ndarray,
    lbd1: float = 5e-6,
    lbd2: float = 0.0,
    k: float = 1,
    mean: float = 0.5,
):
    """Brightness constrained screenened poisson equation for contrast
    enhancement and low-light enhancement.

    Parameters
    ----------
    img : np.ndarray
        An RGB or grayscale image with shape `(H, W, *)`.
    lbd1 : float, default=5e-6
        The strength of the constraint to the brightness.
    lbd2 : float, default 0.0
        The strength of the data fidelity term.
    k : float, default=1
        The ratio of contrast enhancement.
    mean : float, default=0.5
        Mean brightness.

    Returns
    -------
    np.ndarray
        Enhanced image with the same shape as input.
    """
    num_ch = 1 if img.ndim == 2 else img.shape[2]
    if num_ch == 3:
        gray = img.max(2, keepdims=True)
    elif num_ch == 1:
        gray = img
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    gray_f = np.fft.rfft2(gray, axes=(0, 1))

    fft_laplacian = -get_freq_laplacian(gray_f, form='continuous', d=1)
    gray_f *= k * fft_laplacian + lbd2
    gray_f[0, 0] += mean * (lbd1 * img.shape[-1] * img.shape[-2])
    gray_f /= fft_laplacian + (lbd1 + lbd2)
    new_gray = np.fft.irfft2(gray_f, s=img.shape[-2:], axes=(0, 1))
    new_gray.clip(0.0, None, out=new_gray)
    new_gray /= gray + 1e-7
    res = img * new_gray
    res.clip(0.0, 1.0, out=res)
    return res
