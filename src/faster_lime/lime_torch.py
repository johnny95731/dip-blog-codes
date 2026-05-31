# Modified from the following code: https://github.com/aeinrw/LIME/
# Based on the MIT License

# MIT License

# Copyright (c) 2020 Wei

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Literal

import torch


def _weighting_strategy(t_hat: torch.Tensor, strategy: int):
    if strategy == 2:
        mat_w = torch.stack(_derivative(t_hat), 0)
        mat_w.abs_().add_(1.0).reciprocal_()
        return mat_w
    else:
        b, _, h, w = t_hat.shape
        return t_hat.new_ones((2, b, 1, h, w))


def _get_freq_laplacian(
    img: torch.Tensor,
    d: float | None = 1.0,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a frequency domain Laplacian filter for rfft image.

    Parameters
    ----------
    img_size : int | tuple[int, int] | torch.Tensor
        The size of rfft image. Shape `[size_y, size_x]`.
    form : {'continuous', '4-neighbor', 'diag'}, default='continuous'
        The form of approximation of discrete Laplacian filter in frequency
        domain.

        - `'continuous'`: Discretize the Fourier transform of the continuous
        Laplacian operator. Better
        - `'5-point'`: Computes the Fourier transform of the 5-point stencil
        Laplacian operator. This can be used to solve the PDE.
        - `'9-point'`: Computes the Fourier transform of the 9-point stencil
        Laplacian operator.
    d : float | None, default=1.0
        The sampling length scale. If None, uses 1 / img_size. For details,
        see `torch.fft.fftfreq` and `torch.fft.rfftfreq`.
    dtype : torch.dtype, default=None
        The Data type of the filter.
    device : torch.device, default=None
        The Device of the returned filter.

    Returns
    -------
    torch.Tensor
        2D Laplacian filter.

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
    _ksize = img.shape[-2:]

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

    # Discretize the Fourier transform of the continuous Laplacian operator
    freq_y.mul_(2 * torch.pi).square_()
    freq_x.mul_(2 * torch.pi).square_()
    fft_laplacian = freq_y.neg_() - freq_x
    return fft_laplacian


def _derivative(
    t_hat: torch.Tensor,
    direction: Literal['y', 'x', 'both'] = 'both',
):
    dy = None
    dx = None
    if direction == 'y' or direction == 'both':
        dy = t_hat.diff(dim=-2, append=t_hat[..., -1:, :])
    if direction == 'x' or direction == 'both':
        dx = t_hat.diff(dim=-1, append=t_hat[..., -1:])
    return dy, dx


def _subproblem_t(
    t_hat: torch.Tensor,
    img_t: torch.Tensor,
    img_g: torch.Tensor,
    img_z: torch.Tensor,
    mu: float,
    # const
    fft_laplacian: torch.Tensor,
    # two: torch.Tensor,
    # # preallocate
    temp: torch.Tensor,
    _numerator: torch.Tensor,
    _denominator: torch.Tensor,
):
    img_x = torch.sub(img_g, img_z, alpha=1 / mu, out=temp)
    dy = img_x[0]
    dx = img_x[1]
    dyx = _derivative(dy, 'x')[1]
    dxy = _derivative(dx, 'y')[0]

    torch.add(dyx, dxy, out=_numerator)
    _numerator.mul_(mu).add_(t_hat, alpha=2.0)
    numerator = torch.fft.rfft2(_numerator)  # type: torch.Tensor

    torch.mul(fft_laplacian, mu, out=_denominator).add_(2.0)
    numerator /= _denominator
    torch.fft.irfft2(numerator, s=img_t.shape[-2:], out=img_t)  # type: torch.Tensor
    mini = img_t.amin((-1, -2), keepdim=True)
    maxi = img_t.amax((-1, -2), keepdim=True)
    img_t.sub_(mini).div_(maxi.sub_(mini))


def _subproblem_g(
    dt: torch.Tensor,
    img_g: torch.Tensor,
    img_z: torch.Tensor,
    mu: float,
    mat_w: torch.Tensor,
    alpha: float,
    tempw: torch.Tensor,
):
    epsilon = torch.mul(mat_w, alpha / mu, out=tempw)

    x = torch.add(dt, img_z, alpha=1 / mu, out=img_g)
    sign = torch.sign(x)
    img_g.abs_().sub_(epsilon).relu_().mul_(sign)
    return x


def _subproblem_z(
    dt: torch.Tensor,
    mat_g: torch.Tensor,
    mat_z: torch.Tensor,
    mu: float,
):
    mat_z.add_(dt, alpha=mu).sub_(mat_g, alpha=mu)
    return mat_z


def apply_lime(
    img: torch.Tensor,
    num_iter: int = 5,
    alpha: float = 2,
    rho: float = 2,
    gamma: float = 0.5,
    strategy=2,
):
    """An implements of Guo's work [1]. The papper is availible on arxiv [2].

    Parameters
    ----------
    img : torch.Tensor
        Image in the range of [0, 1] with shape (B, C, H, W).
    num_iter : int, optional
        Maximum number of iterations, by default 10
    alpha : float, optional
        Multipler in subproblem G, by default 2
    rho : float, optional
        Multipler for mu, by default 2
    gamma : float, optional
        Gamma correction the esimated illuminant befoe enhancing,
        by default 0.5.
    strategy : int, optional
        Strategy for initlizing matrix W, by default 2.

    Returns
    -------
    torch.Tensor
        Enhanced image. shape (H, W, C). dtype uint8.

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    assert 3 <= img.ndim <= 4
    is_not_batch = img.ndim == 3
    if is_not_batch:
        img = img.unsqueeze(0)
    b, c, h, w = img.shape
    # constants
    t_hat = torch.amax(img, -3, keepdim=True)
    dtype = img.dtype if img.is_floating_point() else torch.float32
    device = img.device
    fft_laplacian = -_get_freq_laplacian(img, d=1, dtype=dtype, device=device)
    mat_w = _weighting_strategy(t_hat, strategy)
    # iteration variables
    img_t = img.new_zeros((b, 1, h, w), dtype=dtype)
    img_g = img.new_zeros((2, b, 1, h, w), dtype=dtype)
    img_z = img.new_zeros((2, b, 1, h, w), dtype=dtype)
    mu = 1.0
    # pre-allocate variables
    tempt = img_t.new_empty((b, 1, h, w))
    tempg = img_g.new_empty((2, b, 1, h, w))
    templap = torch.empty_like(fft_laplacian, dtype=dtype)
    # two = torch.full_like(fft_laplacian, 2.0, dtype=dtype)

    _subproblem_t(
        t_hat,
        img_t,
        img_g,
        img_z,
        mu,
        fft_laplacian,
        # two,
        tempg,
        tempt,
        templap,
    )
    for _ in range(num_iter - 1):
        dt = torch.stack(_derivative(img_t), 0)
        _subproblem_g(dt, img_g, img_z, mu, mat_w, alpha, tempg)
        _subproblem_z(dt, img_g, img_z, mu)
        mu *= rho
        _subproblem_t(
            t_hat,
            img_t,
            img_g,
            img_z,
            mu,
            fft_laplacian,
            # two,
            tempg,
            tempt,
            templap,
        )
    img_t.relu_()
    img_t **= gamma
    res = img.div(img_t.add_(1e-8))
    if is_not_batch:
        res.squeeze_(0)
    return res
