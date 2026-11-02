"""Color balance functions including chromatic adaptation transform, gray world
algorithm, etc...
"""

import math
from typing import Literal

import torch
from torch.nn import functional as F


# Utils
def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    device = rgb.device
    # All chrominance channel are normalized to the range [-0.5, 0.5].
    matrix = torch.tensor(((0.299, 0.587, 0.114),), device=device)
    gray = torch.einsum('...oc,...chw->...ohw', matrix, rgb)
    return gray


def histogram(
    img: torch.Tensor,
    bins: int = 256,
    density: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the histogram of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image in the range of [0, 1] with shape `(*, H, W)`
    bins : int, default=256
        The number of groups in data range.
    density : bool, default=False
        If true, return the pdf of each channel.

    Returns
    -------
    hist : torch.Tensor
        The histogram or density. Shape `(*, bins)`.
    img_idx : torch.Tensor
        The index of histogram for each pixel. Shape `(*, H * W)`.
        The `img_idx` is returned only if `ret_index` is true.
    """
    if not isinstance(bins, int):
        raise TypeError(f'`bins` must be an integer: {type(bins)}.')
    idx = (img * (bins - 1)).add_(0.1).clip_(0, bins - 1).long()

    flatted_idx = idx.flatten(start_dim=-2)
    hist = torch.zeros(
        img.shape[:-2] + (bins,),
        dtype=torch.int32,
        device=img.device,
    )
    hist.scatter_add_(
        dim=-1, index=flatted_idx, src=hist.new_ones(1).expand_as(flatted_idx)
    )
    if density:
        num_el = flatted_idx.size(-1)
        hist = hist.float() / num_el
    return hist, idx


def linearize_adobe_rgb(
    adobe_rgb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    linear = torch.empty_like(adobe_rgb) if out is None else out

    torch.pow(adobe_rgb, 2.19921875, out=linear)
    return linear


def gammaize_adobe_rgb(
    linear: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    adobe_rgb = torch.empty_like(linear) if out is None else out

    torch.pow(linear, 1 / 2.19921875, out=adobe_rgb)
    return adobe_rgb


# Filters
def filter2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
    padding: int | list[int],  # (pad_y, pad_x)
    mode: str = 'reflect',
) -> torch.Tensor:
    is_single_image = img.ndim == 3
    if is_single_image:
        img = img.unsqueeze(0)
    num_ch = img.size(-3)
    if not torch.is_floating_point(img):
        img = img.float()
    # kernel shape
    if kernel.ndim == 1:
        kernel = kernel.view(1, 1, 1, kernel.shape[0])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 2:
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(num_ch, 1, 1, 1)
    elif kernel.ndim == 3:
        kernel = kernel.unsqueeze(1)
    kernel = kernel.contiguous()
    #
    if isinstance(padding, int):
        padding = (padding,) * 4
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    #
    if padding is None:
        res = F.conv2d(img, weight=kernel, groups=num_ch)
    elif mode == 'constant' and (
        padding[0] == padding[1] and padding[2] == padding[3]
    ):
        res = F.conv2d(img, weight=kernel, padding=padding[::2], groups=num_ch)
    else:
        padding = [*padding[2:4], *padding[:2]]
        img = F.pad(img, pad=padding, mode=mode)
        res = F.conv2d(img, weight=kernel, groups=num_ch)
    if is_single_image:
        res = res.squeeze_(0)
    return res


def get_gaussian_kernel(
    ksize: int = 5,
    sigma: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    if ksize <= 0 and sigma <= 0:
        raise ValueError(f'ksize and sigma can not be both non-positive.')
    elif ksize <= 0:
        ksize = max(int(6 * sigma + 1), 3) | 1
    elif sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    half = ksize // 2
    kernel1d = torch.linspace(-half, half, ksize)
    kernel1d.div_(sigma).square_().mul_(-0.5).exp_()
    if normalize:
        kernel1d /= torch.sum(kernel1d)
    kernel2d = torch.outer(kernel1d, kernel1d)
    return kernel2d


def gaussian_blur(
    img: torch.Tensor,
    ksize: int | tuple[int, int] = 3,
    sigma: float | tuple[float, float] = 0.0,
    mode: str = 'reflect',
) -> torch.Tensor:
    kernel = get_gaussian_kernel(ksize, sigma, True).to(img.device)
    padding = ksize // 2
    bluured = filter2d(img, kernel, padding=padding, mode=mode)
    return bluured


def laplacian(
    img: torch.Tensor,
    mode: str = 'reflect',
) -> torch.Tensor:
    dtype = img.dtype if img.is_floating_point() else torch.float32
    kernel = torch.tensor(
        (
            (0, -1, 0),
            (-1, 4, -1),
            (0, -1, 0),
        ),
        dtype=dtype,
        device=img.device,
    )
    grad = filter2d(img, kernel, (1, 1, 1, 1), mode)
    return grad


# Correction functions
def balance_in_rgb(
    rgb: torch.Tensor,
    illum: torch.Tensor,
    ch_gain: Literal['green', 'bri', 'mean'] = 'green',
):
    """White balance by a given illuminant color in RGB space.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image with shape `(*, 3, H, W)`.
    illum : torch.Tensor
        Illuminant color of image. Shape `(*, 3, H or 1, W or 1)`.
    ch_gain : {"green", "bri", "mean"}, default="green"
        The formula of channel gain.

    Returns
    -------
    torch.Tensor
        A balanced image with shape `(*, C, H, W)`.
    """
    assert ch_gain in ('green', 'bri', 'mean'), (
        f'`ch_gain` must be one of ("green", "bri", "mean"), but got {ch_gain}.'
    )

    if ch_gain == 'green':
        gain = illum[..., 1:2, :, :] / illum
    elif ch_gain == 'bri':
        gain = rgb_to_gray(illum) / illum
    elif ch_gain == 'mean':
        gain = illum.mean(-3, keepdim=True) / illum

    res = rgb * gain
    return res


def simplest_color_balance(
    img: torch.Tensor,
    dark_percent: float = 0.01,
    light_percent: float = 0.01,
):
    """Clip top-k1 and bottom-k2 percentage values and normalize to `[0, 1]`.
    The algorithm is proposed by Limare et al [1].

    This function will not estimate the illuminant.

    Parameters
    ----------
    img : torch.Tensor
        An image with shape `(*, C, H, W)`.
    dark_percent : float, default=0.01
        The percentage value for clipping the lowest `dark_percent`% values.
    light_percent : float, default=0.01
        The percentage value for clipping the highest `light_percent`% values.

    Returns
    -------
    torch.Tensor
        A balanced image with shape `(*, C, H, W)`.

    References
    ----------
    [1] Nicolas Limare, Jose-Luis Lisani, Jean-Michel Morel, Ana Belén Petro, and Catalina Sbert, Simplest Color Balance, Image Processing On Line, 1 (2011), pp. 297-315. https://doi.org/10.5201/ipol.2011.llmps-scb
    """
    bins = 256
    x = torch.linspace(0, 1, bins, device=img.device)
    hist, _ = histogram(img, bins=bins)
    cdf = hist.cumsum(-1)  # shape: [..., bins]
    cdf = cdf / cdf[..., -1:]
    # Index of percentile
    maxi_idx = (cdf < (1 - light_percent)).count_nonzero(-1).clip_(0, bins - 1)
    mini_idx = (cdf < dark_percent).count_nonzero(-1).clip_(0, bins - 1)
    #
    mini = torch.take(x, index=mini_idx).unsqueeze_(-1).unsqueeze_(-1)
    maxi = torch.take(x, index=maxi_idx).unsqueeze_(-1).unsqueeze_(-1)
    balanced = (img - mini).div_((maxi - mini))
    balanced = balanced.clip(0.0, 1.0)
    return balanced


# Illuminant color estimation
def general_gray_world(
    rgb: torch.Tensor,
    p: float = 1.0,
    sigma: float = 0.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> torch.Tensor:
    """White balance by the General Gray-World assumption.

    .. math::
        color = (mean(Red), mean(Green), mean(Blue))

    If `est_only` is `True`, the function only estimate illuminant
    color without applying white balance.

    Parameters
    ----------
    rgb : torch.Tensor
        A non-negative RGB Image with shape `(*, C, H, W)`.
    p : int | float, default=1.0
        The exponent parameter of p-norm.
    sigma : int | float, default=0.0
        The blurrness parameter.
    est_only : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.
    is_linear : bool, default=False
        Whether `rgb` is linear RGB. If `False` is given, `rgb` will be
        converted to linear RGB before estimating.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `est_only` is
        `False`.
    illum : torch.Tensor
        Estimated illuminant color (linear RGB) with shape `(*, C, 1, 1)` when
        `est_only` is `True`.
    """
    assert isinstance(p, (int, float)) and 0 < p < torch.inf
    assert isinstance(sigma, (int, float)) and sigma >= 0
    lrgb = rgb if is_linear else linearize_adobe_rgb(rgb)
    _lrgb = (  # General Gray world: blurrness + p-norm
        lrgb if abs(sigma) < 1e-7 else gaussian_blur(lrgb, 0, sigma)
    )
    # Estimation
    if abs(1 - p) < 1e-7:
        illum = _lrgb.mean((-1, -2), keepdim=True)
    else:
        illum = _lrgb.pow(p).mean((-1, -2), keepdim=True).pow_(1 / p)

    if est_only:
        return illum
    balanced = balance_in_rgb(lrgb, illum, ch_gain='green').clip_(0.0, 1.0)
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    return balanced


def gray_edge(
    rgb: torch.Tensor,
    edge: torch.Tensor | None = None,
    p: float = 6.0,
    sigma: float = 5.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> torch.Tensor:
    """White balance by the gray-edge algorithm.

    .. math::
        color = (mean(diff(Red)), mean(diff(Green)), mean(diff(Blue)))

    If `est_only` is `True`, the function only estimate illuminant
    color without applying white balance.

    Parameters
    ----------
    rgb : torch.Tensor
        Image in RGB space with shape `(*, C, H, W)`.
    edge : torch.Tensor
        The edge of the image with shape `(*, C, H, W)`.
    p : int | float, default=6.0
        The exponent parameter of p-norm.
    sigma : int | float, default=5.0
        The blurrness parameter.
    est_only : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.
    is_linear : bool, default=False
        Whether `rgb` is linear RGB. If `False` is given, `rgb` will be
        converted to linear RGB before estimating.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `est_only` is
        `False`.
    illum : torch.Tensor
        Estimated illuminant color (linear RGB) with shape `(*, C, 1, 1)` when
        `est_only` is `True`.
    """
    assert isinstance(p, (int, float)) and p > 0
    assert isinstance(sigma, (int, float)) and sigma >= 0
    if edge is None:
        edge = laplacian(rgb).abs_()
    elif isinstance(edge, torch.Tensor):
        edge = edge.abs()
    else:
        raise TypeError(
            f'`edge` must be "None" or a tensor, but got "{type(edge)}"'
        )
    if math.isinf(p):
        illum = white_patch_balance(
            edge,
            sigma=sigma,
            est_only=True,
            is_linear=is_linear,
        )
    else:
        illum = general_gray_world(
            edge,
            p=p,
            sigma=sigma,
            est_only=True,
            is_linear=is_linear,
        )

    if est_only:
        return illum
    lrgb = rgb if is_linear else linearize_adobe_rgb(rgb)
    balanced = balance_in_rgb(lrgb, illum, ch_gain='green').clip_(0.0, 1.0)
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    return balanced


def white_patch_balance(
    rgb: torch.Tensor,
    q: int | float | torch.Tensor = 1.0,
    sigma: float = 0.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> torch.Tensor:
    """White balance by generalized white patch algorithm.

    .. math::
        color = (quantile(Red, q), quantile(Gerrn, q), quantile(Blue, q))

    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB Image in range of [0, 1] with shape `(*, C, H, W)`.
        If ndim > 3, the quantile value is calculated across images,
        and images will be scaled by same factors.
    q : int | float | torch.Tensor, default=1.0
        q-quantile. The values will be cliped to [0, 1].
        - A single number: the quantile for all channels.
        - Tensor with shape `(3,)`: the quantiles of channels.
    sigma : int | float, default=0.0
        The blurrness parameter.
    est_only : bool, default=False
        Return the estimated illuminant color instead of returning the
        balanced image.
    is_linear : bool, default=False
        Whether `rgb` is linear RGB. If `False` is given, `rgb` will be
        converted to linear RGB before estimating.

    Returns
    -------
    balanced : torch.Tensor
        A balanced image with the shape `(*, C, H, W)` when `est_only` is
        `False`.
    illum : torch.Tensor
        Estimated illuminant color with shape `(*, C, 1, 1)` when
        `est_only` is `True`.
    """
    assert isinstance(sigma, (int, float)) and sigma >= 0
    is_not_batch = rgb.ndim == 3
    if is_not_batch:
        rgb = rgb.unsqueeze(0)

    lrgb = rgb if is_linear else linearize_adobe_rgb(rgb)
    _lrgb = lrgb if abs(sigma) < 1e-7 else gaussian_blur(lrgb, 0, sigma)

    bins = 256
    x = torch.linspace(0, 1, bins, device=rgb.device)
    hist, _ = histogram(_lrgb, bins=bins)
    cdf = hist.cumsum(-1)  # shape: [..., bins]
    cdf = cdf / cdf[..., -1:]

    shape = rgb.shape[:-2]
    if isinstance(q, float):  # Two if-else for `torch.jit.script``
        q = torch.full(shape, q, device=rgb.device)
    elif isinstance(q, int):
        q = torch.full(shape, q, device=rgb.device)
    else:
        q = q.to(rgb.device).broadcast_to(shape)
    q = q.clip(0, 1).unsqueeze_(-1)

    index = (cdf < q).count_nonzero(-1).unsqueeze_(-1).unsqueeze_(-1)
    illum = torch.take(x, index=index)
    if est_only:
        if is_not_batch:
            illum = illum.squeeze_(0)
        return illum

    balanced = balance_in_rgb(lrgb, illum, ch_gain='green').clip_(0.0, 1.0)
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    if is_not_batch:
        balanced = balanced.squeeze_(0)
    return balanced
