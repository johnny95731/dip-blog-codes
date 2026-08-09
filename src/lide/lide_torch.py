from typing import Literal

import torch
from torch.nn import functional as F


# Utils
def _gaussian_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    z_score = (x - mean).div_(std.mul_(2**0.5))
    res = torch.erf_(z_score).add_(1.0).mul_(0.5)
    return res


def _laplace_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    diff = x - mean
    sign = diff.sign()
    z_score = diff.abs_().div_(std).mul_(-(2**0.5))
    part = z_score.exp_().mul_(sign)
    res = sign.sub_(part).add_(1.0).mul_(0.5)
    return res


def _cauchy_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    z_score = (x - mean).div_(std)
    res = z_score.arctan_().mul_(1 / torch.pi).add_(0.5)
    return res


def _logistic_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    z_score = (x - mean).div_(std)
    res = z_score.sigmoid_()
    return res


def _hyperbolic_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    z_score = (x - mean).div_(std)
    res = z_score.mul_(torch.pi / 2).exp_().arctan_().mul_(2 / torch.pi)
    return res


def convex_comb(x: torch.Tensor, y: torch.Tensor, c: float):
    """res = c * x + (1-c) * y"""
    if abs(c) < 1e-10:
        res = y
    elif abs(1 - c) < 1e-10:
        res = x
    else:
        res = c * x + (1 - c) * y
    return res


#
def modified_lide(
    rgb: torch.Tensor,
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
    rgb : torch.Tensor
        An RGB or grayscale image with shape `(*, C, H, W)`.
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
    torch.Tensor
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
    num_ch = rgb.size(-3)
    if num_ch == 3:
        gray = rgb.mean(-3, keepdim=True)
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    gray = gray.add(1e-8)
    #
    std, mean = torch.std_mean(gray, dim=(-1, -2), keepdim=True)
    if radius is not None:
        ksize = 2 * radius + 1
        local_mean = F.avg_pool2d(
            gray,
            ksize,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        sq_mean = F.avg_pool2d(
            gray.square(),
            ksize,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        # std(x) = mean(x**2) - mean(x)**2
        local_std = sq_mean.sub_(mean.square())

        mean = convex_comb(mean, local_mean, eta_m)
        std = convex_comb(std, local_std, eta_s)
    #
    std = std.clip_(std_min, std_max)
    center = mean.sub_(std, alpha=alpha) if abs(alpha) > 1e-10 else mean

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
    if num_ch == 3 and use_gamma:
        res = rgb * (res / gray).pow(gamma)
    elif num_ch == 3:
        res = rgb * res / gray
    elif use_gamma:
        res = res.pow(gamma)
    res.clip_(0.0, 1.0)
    return res
