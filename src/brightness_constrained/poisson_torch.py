from typing import Literal

import torch


# utilities
def __default_dtype(x: torch.Tensor) -> torch.dtype:
    dtype = x.dtype if torch.is_floating_point(x) else torch.float32
    return dtype


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


# Main functions
def bri_constrained_poisson(
    img: torch.Tensor,
    lbd1: float = 5e-6,
    lbd2: float = 0.0,
    k: float = 1,
    mean: float = 0.5,
):
    """Brightness constrained screenened poisson equation for contrast
    enhancement and low-light enhancement.

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
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
    torch.Tensor
        Enhanced image with the same shape as input.
    """
    dtype = __default_dtype(img)
    gray = torch.mean(img, -3, keepdim=True)
    gray_f = torch.fft.rfft2(gray)

    fft_laplacian = get_freq_laplacian(
        gray_f, form='continuous', d=1, dtype=dtype, device=img.device
    ).neg_()
    gray_f.mul_(k * fft_laplacian + lbd2)
    gray_f[..., 0, 0] += mean * (lbd1 * img.shape[-1] * img.shape[-2])
    gray_f = gray_f.div_(fft_laplacian.add_(lbd1 + lbd2))
    new_gray = torch.fft.irfft2(gray_f, s=img.shape[-2:])
    new_gray.relu_()
    new_gray.div_(gray.add_(1e-7))
    res = img.mul(new_gray)
    return res
