import math
from typing import Literal

import cv2
import numpy as np
from numba import jit, types

from .noise_np import (
    _energy_mask,
    generate_gaussian_noise,
    generate_uniform_noise,
)


@jit([
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A', readonly=True),
        types.int64,
    ),
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A', readonly=True),
        types.Omitted(4),
    ),
])
def _quantization(
    img: np.ndarray,
    num_colors: int = 4,
):
    # Map intensity to quantization intensity
    q_map = np.empty(256, dtype=np.uint8)
    start = 0
    for i in range(num_colors):
        intensity = 255 / (num_colors - 1) * i
        end = int(255 / num_colors * (i + 1))
        q_map[start : end + 1] = intensity
        start = end
    #
    res = np.empty_like(img, dtype=np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            for c, val in enumerate(pixel):
                index = min(max(int(round(val)), 0), 255)
                res[y, x, c] = q_map[index]
    return res


def quantization(img: np.ndarray, num_colors: int = 4):
    """Quantization the image to given number of colors.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W)` or `(H, W, C)`. The image will be convert
        to uint8 before quantization.
    num_colors : int, default44
        Number of colors

    Returns
    -------
    np.ndarray
        Quantization image.
    """
    assert isinstance(num_colors, int)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    is_grayscale = img.ndim == 2
    if is_grayscale:
        img = img[..., None]
    res = _quantization(img, num_colors)
    if is_grayscale:
        res = res.squeeze(2)
    return res


def create_bayer_matrix(order: int = 4):
    """Create an nxn Bayer matrix, where n=order"""
    assert isinstance(order, int) and order > 0, (
        '`order` must be a positive interger.'
    )
    assert 2 ** math.log2(order) == order
    if order == 2:
        bayer = np.array(((0, 2), (3, 1)), dtype=np.float32)
        return bayer
    last = create_bayer_matrix(order // 2)
    x4 = last * 4
    top_left = x4
    top_right = x4 + 2
    bottom_left = x4 + 3
    bottom_right = x4 + 1
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    bayer = np.vstack((top, bottom))
    return bayer


def ordered_dithering(img: np.ndarray, order: int = 4):
    """Image dithering by Bayer matrix with order `order`.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W)` or `(H, W, C)`. The image will be convert
        to uint8 before dithering.
    order : int, default=4
        Order of the Bayer matrix.

    Returns
    -------
    np.ndarray
        Output binary image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.float32, copy=False)
        img = cv2.convertScaleAbs(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        )
    bayer = create_bayer_matrix(order) * (255 / order**2)
    # expansion
    shape = img.shape
    exp_y = math.ceil(shape[0] / order)
    exp_x = math.ceil(shape[1] / order)
    bayer = np.tile(bayer, (exp_y, exp_x))[: shape[0], : shape[1]]
    if img.ndim == 3:
        bayer = bayer[..., None]
    res = (img > bayer).astype(np.uint8, copy=False)
    res *= 255
    return res


@jit([
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A', readonly=True),
        types.Array(types.float32, 2, 'A', readonly=True),
        types.int64,
    ),
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A', readonly=True),
        types.Array(types.float32, 2, 'A', readonly=True),
        types.Omitted(4),
    ),
])
def _error_diffusion(
    img: np.ndarray,
    matrix: np.ndarray,
    num_colors: int = 4,
):
    """Performs error diffusion with the given weight matrix.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)` and dtype `uint8`.
    matrix : np.ndarray
        2D weight matrix. The last zero will be treated as center, point that
        is handling.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    h, w = img.shape[:2]
    # For mapping intensity to quantization intensity
    q_map = np.empty(256, dtype=np.float32)
    start = 0
    for i in range(num_colors):
        intensity = 255 / (num_colors - 1) * i
        end = int(255 / num_colors * (i + 1))
        q_map[start : end + 1] = intensity
        start = end
    # Find center in weight matrix
    cy = 0
    cx = 0
    for u, mat_row in enumerate(matrix):
        for v, val in enumerate(mat_row):
            if val != 0:
                cy = u
                cx = v - 1
                break
        if cy != 0 or cx != 0:
            break
    if cx < 0:
        cy -= 1
        cx = matrix.shape[1] - 1
    #
    res = img.astype(np.float32)
    for y, row in enumerate(res):
        for x, pixel in enumerate(row):
            for c, val in enumerate(pixel):
                index = min(max(round(val), 0), 255)
                new_val = q_map[index]
                error = np.float32(val) - new_val
                res[y, x, c] = np.uint8(new_val)
                # Diffuse the error to other pixels.
                for v, mat_row in enumerate(matrix):
                    v += y - cy
                    if v == h:
                        break
                    for u, weight in enumerate(mat_row):
                        u += x - cx
                        if weight > 0 and 0 <= u < w:
                            res[v, u, c] += error * weight
    return res.astype(np.uint8)


@jit([
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A'), types.int64
    ),
    types.Array(types.uint8, 3, 'C')(
        types.Array(types.uint8, 3, 'A'), types.Omitted(4)
    ),
])
def _floyd_steinberg(img: np.ndarray, num_colors: int = 4):
    """An example code of the Floyd-Steinberg dithering. A spesific code
    will faster than general code.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)` and dtype `uint8`.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    index_map = np.empty(256, dtype=np.float32)
    start = 0
    for i in range(num_colors):
        intensity = 255 / (num_colors - 1) * i
        end = int(255 / num_colors * (i + 1))
        index_map[start : end + 1] = intensity
        start = end
    h, w = img.shape[:2]
    res = img.astype(np.float32)
    for y, row in enumerate(res):
        for x, pixel in enumerate(row):
            for c, val in enumerate(pixel):
                index = min(max(round(val), 0), 255)
                new_val = index_map[index]
                error = (val - new_val) / 16
                res[y, x, c] = new_val
                if x + 1 < w:
                    res[y, x + 1, c] += error * 7
                if y + 1 < h:
                    if x - 1 >= 0:
                        res[y + 1, x - 1, c] += error * 3
                    res[y + 1, x, c] += error * 5
                    if x + 1 < w:
                        res[y + 1, x + 1, c] += error
    return res.astype(np.uint8)


def floyd_steinberg_dithering(img: np.ndarray, num_colors: int = 4):
    """Image dithering by Floyd-Steinberg's weight.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to uint8 before dithering.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.float32, copy=False)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    is_grayscale = img.ndim == 2
    if is_grayscale:
        img = img[..., None]
    matrix = np.array(((0, 0, 7), (3, 5, 1)), dtype=np.float32) / 16
    res = _error_diffusion(img, matrix, num_colors)
    if is_grayscale:
        res = res.squeeze(2)
    return res


def atkinson_dithering(img: np.ndarray, num_colors: int = 4):
    """Image dithering by Atkinson's weight.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to uint8 before dithering.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.float32, copy=False)
        img = cv2.convertScaleAbs(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        )
    is_grayscale = img.ndim == 2
    if is_grayscale:
        img = img[..., None]
    matrix = (
        np.array(((0, 0, 1, 1), (1, 1, 1, 0), (0, 1, 0, 0)), dtype=np.float32)
        / 8
    )
    res = _error_diffusion(img, matrix, num_colors)
    if is_grayscale:
        res = res.squeeze(2)
    return res


def jarvis_judice_ninke_dithering(img: np.ndarray, num_colors: int = 4):
    """Image dithering by Jarvis-judice-Ninke's weight.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to uint8 before dithering.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.float32, copy=False)
        img = cv2.convertScaleAbs(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        )
    is_grayscale = img.ndim == 2
    if is_grayscale:
        img = img[..., None]
    matrix = (
        np.array(
            ((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)),
            dtype=np.float32,
        )
        / 48
    )
    res = _error_diffusion(img, matrix, num_colors)
    if is_grayscale:
        res = res.squeeze(2)
    return res


def burkes_dithering(img: np.ndarray, num_colors: int = 4):
    """Image dithering by Burkes's weight.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to uint8 before dithering.
    num_colors : int, defaul=4
        Number of colors

    Returns
    -------
    np.ndarray
        Output image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.float32, copy=False)
        img = cv2.convertScaleAbs(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        )
    is_grayscale = img.ndim == 2
    if is_grayscale:
        img = img[..., None]
    matrix = (
        np.array(
            ((0, 0, 0, 8, 4), (2, 4, 8, 4, 2)),
            dtype=np.float32,
        )
        / 32
    )
    res = _error_diffusion(img, matrix, num_colors)
    if is_grayscale:
        res = res.squeeze(2)
    return res


# Random dithering
def generate_power_noise(
    shape: tuple[int, ...],
    beta: float = 1,
    noise_type: Literal['g', 'u'] = 'u',
) -> np.ndarray[tuple[int, ...], np.float32]:
    """Generate a 2D power-law noise with exponent parameter `beta`. The
    frequency energy is proportional to `frequency**(-beta)`

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of image, must be `(H, W)` or `(H, W, C)`
    beta : float, default=1
        The exponent parameter.
    noise_type : {'g', 'u'}, default='u'
        Gaussian noise ('g') or uniform noise ('u').

    Returns
    -------
    np.ndarray
        Power-law uniform noise with the given shape.
    """
    assert 2 <= len(shape) <= 3, (
        '`shape` must be `(height, width)` or `(height, width, channel)`.'
    )
    shape = list(shape)
    h, w = shape[:2]
    mask = _energy_mask(shape, exponent=-beta, dtype=np.float32)
    if len(shape) == 3:
        mask = mask[..., None]
    # noise in frequency domain
    if noise_type == 'u':
        noise = generate_uniform_noise(shape, 0, 1)
        noisef = np.fft.rfft2(noise, axes=(0, 1))
    elif noise_type == 'g':
        shape[:2] = mask.shape[:2]
        noisef = generate_gaussian_noise((*shape, 2), 0, 0.1)
        noisef = noisef.view(np.complex64).squeeze(-1)
    noisef *= mask
    #
    noise = np.fft.irfft2(noisef, axes=(0, 1))
    if noise.shape[:2] != (h, w):
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    if noise_type == 'u':
        noise = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)
    elif noise_type == 'g':
        mean_, std_ = [arr.flatten() for arr in cv2.meanStdDev(noise)]
        noise -= mean_
        noise *= 0.2 / std_
        noise += 0.5
    return noise


def power_random_dithering(
    img: np.ndarray,
    beta: float = -1,
    noise_type: Literal['g', 'u'] = 'u',
    mask_size: int | None = None,
):
    """Random dithering by power-law noise.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to float32 in the range of `[0, 1]`.
    beta : float, default=-1
        The exponent parameter.
    noise_type : Literal["g", "u"], default='u'
        Gaussian noise ("g", μ=0.5, σ=0.2) or uniform noise ("u").
    mask_size : int | None, default=None
        The mask size. If `None` is provided, the mask size will equal to
        image size. Otherwise, generates noise mask and expand to the shape
        of image.

    Returns
    -------
    np.ndarray
        Binary image. Dtype `float32`. Value {0, 1}.
    """
    assert noise_type in ('g', 'u')
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    m = [mask_size] * 2 if mask_size is not None else img.shape
    mask = generate_power_noise(m, beta=beta, noise_type=noise_type)
    if mask_size is not None:  # Expand
        shape = img.shape
        exp_y = math.ceil(shape[0] / mask_size)
        exp_x = math.ceil(shape[1] / mask_size)
        mask = np.tile(mask, (exp_y, exp_x))[: shape[0], : shape[1]]
    res = (img > mask).astype(np.float32)
    return res


def white_noise_dithering(
    img: np.ndarray,
    noise_type: Literal['g', 'u'] = 'u',
    mask_size: int | None = None,
):
    """Random dithering by white noise.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to float32 in the range of `[0, 1]`.
    noise_type : Literal["g", "u"], default='u'
        Gaussian noise ("g", μ=0.5, σ=0.2) or uniform noise ("u").
    mask_size : int | None, default=None
        The mask size. If `None` is provided, the mask size will equal to
        image size. Otherwise, generates noise mask and expand to the shape
        of image.

    Returns
    -------
    np.ndarray
        Binary image. Dtype `float32`. Value {0, 1}.
    """
    assert noise_type in ('g', 'u')
    assert mask_size is None or isinstance(mask_size, int)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    m = [mask_size] * 2 if mask_size is not None else img.shape
    if noise_type == 'u':
        mask = generate_uniform_noise(m, 0, 1)
    elif noise_type == 'g':
        mask = generate_gaussian_noise(m, 0.5, 0.2)
    if mask_size is not None:  # Expand
        shape = img.shape
        exp_y = math.ceil(shape[0] / mask_size)
        exp_x = math.ceil(shape[1] / mask_size)
        mask = np.tile(mask, (exp_y, exp_x))[: shape[0], : shape[1]]
    res = (img > mask).astype(np.float32)
    return res


def blue_noise_dithering(
    img: np.ndarray,
    noise_type: Literal['g', 'u'] = 'u',
    mask_size: int | None = None,
):
    """Random dithering by blue noise.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to float32 in the range of `[0, 1]`.
    noise_type : Literal["g", "u"], default='u'
        Gaussian noise ("g", μ=0.5, σ=0.2) or uniform noise ("u").
    mask_size : int | None, default=None
        The mask size. If `None` is provided, the mask size will equal to
        image size. Otherwise, generates noise mask and expand to the shape
        of image.

    Returns
    -------
    np.ndarray
        Binary image. Dtype `float32`. Value {0, 1}.
    """
    res = power_random_dithering(
        img,
        beta=-1,
        noise_type=noise_type,
        mask_size=mask_size,
    )
    return res


def purple_noise_dithering(
    img: np.ndarray,
    noise_type: Literal['g', 'u'] = 'u',
    mask_size: int | None = None,
):
    """Random dithering by purple noise.

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, C)`. The image will be convert
        to float32 in the range of `[0, 1]`.
    noise_type : Literal["g", "u"], default='u'
        Gaussian noise ("g", μ=0.5, σ=0.2) or uniform noise ("u").
    mask_size : int | None, default=None
        The mask size. If `None` is provided, the mask size will equal to
        image size. Otherwise, generates noise mask and expand to the shape
        of image.

    Returns
    -------
    np.ndarray
        Binary image. Dtype `float32`. Value {0, 1}.
    """
    res = power_random_dithering(
        img,
        beta=-2,
        noise_type=noise_type,
        mask_size=mask_size,
    )
    return res
