"""Color balance functions including chromatic adaptation transform, gray world
algorithm, etc...
"""

import math
from typing import Literal

import cv2
import numpy as np


# Utils
def histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute the histogram of an image.

    Parameters
    ----------
    img : np.ndarray
        An uint8 image with shape `(H, W, C)`
    bins : int, default=256
        The number of groups in data range.

    Returns
    -------
    hist : np.ndarray
        The histogram of each channel. Shape `(C, bins)`.
    """
    num_elment = img.shape[0] * img.shape[1]
    hist = [
        cv2.calcHist([ch], [0], None, [bins], [0, 256]).squeeze(-1) / num_elment
        for ch in cv2.split(img)
    ]
    hist = np.stack(hist, axis=0, dtype=np.float32)
    return hist


def linearize_adobe_rgb(adobe_rgb: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, 256, dtype=np.float32)
    table = cv2.pow(x, 2.19921875)
    table = cv2.convertScaleAbs(table, alpha=255)
    linear = cv2.LUT(adobe_rgb, table)
    return linear


def gammaize_adobe_rgb(linear: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, 256, dtype=np.float32)
    table = cv2.pow(x, 1 / 2.19921875)
    table = cv2.convertScaleAbs(table, alpha=255)
    adobe_rgb = cv2.LUT(linear, table)
    return adobe_rgb


def convert_to_8u(img: np.ndarray):
    if np.issubdtype(img.dtype, np.floating):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif not np.issubdtype(img.dtype, np.uint8):
        raise TypeError(f'Invalid dtype `img.dtype`: {img.dtype}')
    return img


# Correction functions
def balance_in_rgb(
    rgb: np.ndarray,
    illum: np.ndarray,
    ch_gain: Literal['green', 'bri', 'mean'] = 'green',
) -> np.ndarray:
    """White balance by a given illuminant color in RGB space.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB image with shape `(H, W, 3)`.
    illum : np.ndarray
        Illuminant color of image. Shape `(3,)`.
    ch_gain : {"green", "bri", "mean"}, default="green"
        The formula of channel gain.

    Returns
    -------
    np.ndarray
        A balanced image with shape `(H, W, 3)`.
    """
    assert ch_gain in ('green', 'bri', 'mean'), (
        f'`ch_gain` must be one of ("green", "bri", "mean"), but got {ch_gain}.'
    )

    if ch_gain == 'green':
        gain = illum[1] / illum
    elif ch_gain == 'bri':
        gain = (
            cv2.cvtColor(illum.reshape(1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(1)
            / illum
        )
    elif ch_gain == 'mean':
        gain = illum.mean() / illum
    #
    if np.issubdtype(rgb.dtype, np.uint8):
        res = []
        for i, ch in enumerate(cv2.split(rgb)):
            res.append(cv2.convertScaleAbs(ch, alpha=gain[i].item()))
        res = cv2.merge(res)
    else:
        res = rgb * gain
    return res


def simplest_color_balance(
    img: np.ndarray,
    dark_percent: float = 0.01,
    light_percent: float = 0.01,
):
    """Clip top-k1 and bottom-k2 percentage values and normalize to `[0, 1]`.
    The algorithm is proposed by Limare et al [1].

    This function will not estimate the illuminant.

    Parameters
    ----------
    img : np.ndarray
        An image with shape `(H, W, C)`.
    dark_percent : float, default=0.01
        The percentage value for clipping the lowest `dark_percent`% values.
    light_percent : float, default=0.01
        The percentage value for clipping the highest `light_percent`% values.

    Returns
    -------
    np.ndarray
        A balanced image with shape `(H, W, C)`.

    References
    ----------
    [1] Nicolas Limare, Jose-Luis Lisani, Jean-Michel Morel, Ana Belén Petro, and Catalina Sbert, Simplest Color Balance, Image Processing On Line, 1 (2011), pp. 297-315. https://doi.org/10.5201/ipol.2011.llmps-scb
    """
    is_uint8 = np.issubdtype(img.dtype, np.uint8)
    img = convert_to_8u(img)

    bins = 256
    x = np.linspace(0, 255, bins, dtype=np.float32)
    hist = histogram(img, bins=bins)
    cdf = hist.cumsum(-1)  # shape: [..., bins]
    cdf = cdf / cdf[..., -1:]
    # Index of percentile
    maxi_idx = cdf < (1 - light_percent)
    maxi_idx = np.count_nonzero(maxi_idx, axis=-1)
    maxi_idx = np.clip(maxi_idx, 0, bins - 1)
    mini_idx = cdf < dark_percent
    mini_idx = np.count_nonzero(mini_idx, axis=-1)
    mini_idx = np.clip(mini_idx, 0, bins - 1)
    #
    mini = np.take(x, mini_idx)[..., None]
    maxi = np.take(x, maxi_idx)[..., None]
    table = (x[None] - mini) / (maxi - mini) * 255
    table = np.clip(table, 0.0, 255.0).astype(np.uint8)

    balanced = []
    for i, ch in enumerate(cv2.split(img)):
        balanced.append(cv2.LUT(ch, table[i]))
    balanced = cv2.merge(balanced)
    if not is_uint8:  # Compress to [0, 1]
        balanced = cv2.LUT(balanced, np.linspace(0, 1, 256, dtype=np.float32))
    return balanced


# Illuminant color estimation
def general_gray_world(
    rgb: np.ndarray,
    p: float = 1.0,
    sigma: float = 0.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> np.ndarray:
    """White balance by the General Gray-World assumption.

    .. math::
        color = (mean(Red), mean(Green), mean(Blue))

    If `est_only` is `True`, the function only estimate illuminant
    color without applying white balance.

    Parameters
    ----------
    rgb : np.ndarray
        A non-negative RGB Image with shape `(H, W, C)`.
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
    balanced : np.ndarray
        A balanced image with the shape `(H, W, C)` when `est_only` is
        `False`.
    illum : np.ndarray
        Estimated illuminant color (linear RGB) with shape `(C,)` when
        `est_only` is `True`.
    """
    assert isinstance(p, (int, float)) and 0 < p < np.inf
    assert isinstance(sigma, (int, float)) and sigma >= 0
    is_uint8 = np.issubdtype(rgb.dtype, np.uint8)
    rgb = convert_to_8u(rgb)
    lrgb = rgb if is_linear else linearize_adobe_rgb(rgb)
    _lrgb = (  # General Gray world: blurrness + p-norm
        lrgb if abs(sigma) < 1e-7 else cv2.GaussianBlur(lrgb, (0, 0), sigma)
    )
    # Estimation
    if abs(1 - p) < 1e-7:
        illum = np.mean(_lrgb, (-2, -3))
    else:
        illum = cv2.pow(_lrgb, p).mean((-2, -3))
        illum = cv2.pow(illum, 1 / p)

    if est_only:
        return illum
    balanced = balance_in_rgb(lrgb, illum, ch_gain='green')
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    if not is_uint8:  # Compress to [0, 1]
        balanced = cv2.LUT(balanced, np.linspace(0, 1, 256, dtype=np.float32))
    return balanced


def gray_edge(
    rgb: np.ndarray,
    edge: np.ndarray | None = None,
    p: float = 6.0,
    sigma: float = 5.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> np.ndarray:
    """White balance by the gray-edge algorithm.

    .. math::
        color = (mean(diff(Red)), mean(diff(Green)), mean(diff(Blue)))

    If `est_only` is `True`, the function only estimate illuminant
    color without applying white balance.

    Parameters
    ----------
    rgb : np.ndarray
        Image in RGB space with shape `(H, W, C)`.
    edge : np.ndarray
        The edge of the image with shape `(H, W, C)`.
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
    balanced : np.ndarray
        A balanced image with the shape `(H, W, C)` when `est_only` is
        `False`.
    illum : np.ndarray
        Estimated illuminant color (linear RGB) with shape `(C,)` when
        `est_only` is `True`.
    """
    assert isinstance(p, (int, float)) and p > 0
    assert isinstance(sigma, (int, float)) and sigma >= 0
    is_uint8 = np.issubdtype(rgb.dtype, np.uint8)
    rgb = convert_to_8u(rgb)
    if edge is None:
        edge = cv2.Laplacian(rgb, cv2.CV_32F)
    elif not isinstance(edge, np.ndarray):
        raise TypeError(
            f'`edge` must be "None" or a tensor, but got "{type(edge)}"'
        )
    edge = np.abs(edge)
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
    balanced = balance_in_rgb(lrgb, illum, ch_gain='green')
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    if not is_uint8:  # Compress to [0, 1]
        balanced = cv2.LUT(balanced, np.linspace(0, 1, 256, dtype=np.float32))
    return balanced


def white_patch_balance(
    rgb: np.ndarray,
    q: int | float | np.ndarray = 1.0,
    sigma: float = 0.0,
    est_only: bool = False,
    is_linear: bool = False,
) -> np.ndarray:
    """White balance by generalized white patch algorithm.

    .. math::
        color = (quantile(Red, q), quantile(Gerrn, q), quantile(Blue, q))

    When q = 1.0, it is the standard white patch balance and equivalent to
    balance by scaling for maximum = 1.

    Parameters
    ----------
    rgb : np.ndarray
        An RGB Image in range of [0, 1] with shape `(H, W, C)`.
        If ndim > 3, the quantile value is calculated across images,
        and images will be scaled by same factors.
    q : int | float | np.ndarray, default=1.0
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
    balanced : np.ndarray
        A balanced image with the shape `(H, W, C)` when `est_only` is
        `False`.
    illum : np.ndarray
        Estimated illuminant color with shape `(C,)` when
        `est_only` is `True`.
    """
    assert isinstance(sigma, (int, float)) and sigma >= 0
    is_uint8 = np.issubdtype(rgb.dtype, np.uint8)
    rgb = convert_to_8u(rgb)

    lrgb = rgb if is_linear else linearize_adobe_rgb(rgb)
    _lrgb = lrgb if abs(sigma) < 1e-7 else cv2.GaussianBlur(lrgb, (0, 0), sigma)

    bins = 256
    x = np.linspace(0, 1, bins, device=rgb.device)
    hist = histogram(_lrgb, bins=bins)
    cdf = hist.cumsum(-1)  # shape: [..., bins]
    cdf = cdf / cdf[..., -1:]

    shape = cdf.shape
    if isinstance(q, (int, float)):  # Two if-else for `torch.jit.script``
        q = np.full(shape, q)
    else:
        q = np.broadcast_to(q, shape)
    q = q.clip(0, 1)

    index = np.count_nonzero(cdf < q, -1)
    illum = np.take(x, index)
    if est_only:
        return illum

    balanced = balance_in_rgb(lrgb, illum, ch_gain='green')
    if not is_linear:
        balanced = gammaize_adobe_rgb(balanced)
    if not is_uint8:  # Compress to [0, 1]
        balanced = cv2.LUT(balanced, np.linspace(0, 1, 256, dtype=np.float32))
    return balanced
