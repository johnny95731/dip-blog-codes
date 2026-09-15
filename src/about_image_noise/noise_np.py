import operator
import time
from functools import reduce

import cv2
import numpy as np

cv2.setRNGSeed(int(time.time()))


def mse(img1, img2, axis: tuple[int, ...] | None = None):
    """Computes the mean square error (MSE) of two images."""
    diff = np.subtract(img1, img2, dtype=np.float32)
    diff = np.square(diff, out=diff)
    res = np.mean(diff, axis=axis)
    return res


def psnr(img1, img2, peak: float = 1, axis: tuple[int, ...] | None = None):
    """Computes the peak signal-to-noise ratio (PSNR) of two images."""
    rmse = np.sqrt(mse(img1, img2, axis=axis))
    res = np.log10(peak / rmse) * 20
    return res


# Spatial
def generate_salt_and_pepper(
    shape: tuple[int, ...],
    p_salt: float = 0.05,
    p_pepper: float = 0.05,
    maxi: float = 1,
) -> np.ndarray[tuple[int, ...], np.float32]:
    """Generate salt-and-pepper noise."""
    noise = None
    if p_salt > 0:
        salt = np.random.binomial(1, p_salt, size=shape)
        noise = np.multiply(salt, maxi, dtype=np.float32)
    if p_pepper > 0:
        pepper = np.random.binomial(1, p_pepper, size=shape)
        pepper = np.multiply(pepper, -maxi, dtype=np.float32)
        if noise is None:
            noise = pepper
        else:
            noise += pepper
    if noise is None:
        noise = np.zeros(shape, dtype=np.float32)
    return noise


def shot_noise(
    img: np.ndarray, peak: float = 200
) -> np.ndarray[tuple[int, ...], np.float32]:
    """Generate noisy image with shot noise.

    See https://stackoverflow.com/questions/19289470/adding-poisson-noise-to-an-image

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, *)`. The image will be normalized to [0, 1]
        before add noise.
    peak : float, default=200
        The higher value will decrease the noise.

    Returns
    -------
     np.ndarray
        Noisy image in the range of [0, 1].
    """
    assert isinstance(peak, (int, float)) and peak > 0
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    noisy_img = np.random.poisson(img * peak)
    noisy_img = np.divide(noisy_img, peak, dtype=np.float32)
    noisy_img = np.clip(noisy_img, 0, 1, out=noisy_img)
    return noisy_img


def add_shot_noise(img: np.ndarray, strength: float = 0.1):
    """Add shot noise to the image.

    See https://stackoverflow.com/questions/19289470/adding-poisson-noise-to-an-image

    Parameters
    ----------
    img : np.ndarray
        Image with shape `(H, W, *)`. The image will be normalized to [0, 1]
        before add noise.
    peak : float, default=200
        The higher value will decrease the noise.

    Returns
    -------
     np.ndarray
        Noisy image in the range of [0, 1].
    """
    dtype = img.dtype if np.issubdtype(np.dtype, np.floating) else np.float32
    noise = np.random.normal(0, np.sqrt(img), img.shape)
    noise = noise.astype(dtype, copy=False)
    noisy_img = img + strength * noise
    return noisy_img


# frequency domain
def _energy_mask(
    shape: tuple[int, ...],
    exponent: float,
    dtype: np.floating = np.float32,
) -> np.ndarray:
    """A mask for power-law noise.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of mask. The output shape will match the rfft2.
    exponent : float
        The exponent of the curve `energy = 1 / f**expoennt`.
    dtype : np.floating, default=np.float32
        Data type of noise

    Returns
    -------
    np.ndarray
        Mask. The shape matches the rfft2.
    """
    assert isinstance(exponent, (int, float))
    x_freq = np.fft.rfftfreq(shape[1])
    x_freq = (x_freq / 2**0.5).astype(dtype, copy=False)[None]
    y_freq = np.fft.fftfreq(shape[0])
    y_freq = (y_freq / 2**0.5).astype(dtype, copy=False)[..., None]
    mask = x_freq**2 + y_freq**2
    mask = np.sqrt(mask, out=mask)  # frequency f
    mask[0, 0] = 1  # handle negative expoent.
    # Divid `exponent` by 2 since energy = square(amplitude).
    # And the mask changes directly the amplitude.
    mask = np.power(mask, exponent / 2, out=mask)
    mask[0, 0] = 0
    return mask


def generate_gaussian_noise(
    shape: tuple[int, ...],
    mean: float = 0,
    std: float = 1,
    dtype: np.floating = np.float32,
) -> np.ndarray:
    assert isinstance(mean, (int, float, np.ndarray)), (
        '`mean` must be a number.'
    )
    assert isinstance(std, (int, float, np.ndarray)), (
        '`std` must be a positive number.'
    )
    assert np.issubdtype(dtype, np.floating), '`dtype` must be a floating type.'
    flat_shape = reduce(operator.mul, shape, 1)
    noise = np.empty((1, flat_shape), dtype=dtype)
    noise = cv2.randn(noise, mean, std).reshape(shape)
    return noise


def generate_uniform_noise(
    shape: tuple[int, ...],
    low: float = 0,
    high: float = 1,
    dtype: np.floating = np.float32,
) -> np.ndarray:
    assert isinstance(low, (int, float, np.ndarray)), '`low` must be a number.'
    assert isinstance(high, (int, float, np.ndarray)), (
        '`high` must be a positive number.'
    )
    assert abs(high - low) > 1e-12, '`low` and `high` must be different.'
    assert np.issubdtype(dtype, np.floating), '`dtype` must be a floating type.'
    if low > high:
        low, high = high, low
    flat_shape = reduce(operator.mul, shape, 1)
    noise = np.empty(flat_shape, dtype=dtype).reshape(shape)
    cv2.randu(noise, low, high)
    return noise


def generate_power_noise(
    shape: tuple[int, ...],
    noisef: np.ndarray | float | None = None,
    beta: float = 1,
    jitter_strength: float = 0,
) -> np.ndarray[tuple[int, ...], np.float32]:
    """Generate a 2D power-law noise with exponent parameter `beta`. The
    frequency energy is proportional to `frequency**(-beta)`

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of noise, must be `(H, W)` or `(H, W, C)`
    noisef : np.ndarray | float | None, default=None
        The noise in frequency domain (by using `fft.rfft2`). If `None`
        is provided, the Gaussian noise is applied.
    beta : float, default=1
        The exponent parameter.
    jitter_strength: float, default=0
        Jitter the mask. Add extra uniform noise `[1-strength, 1+strength]`.

    Returns
    -------
    np.ndarray
        Power-law Gaussian noise with the given shape.
    """
    assert 2 <= len(shape) <= 3, '`shape` must be `(H, W)` or `(H, W, C)`.'
    shape = list(shape)
    h, w = shape[:2]
    mask = _energy_mask(shape, exponent=-beta, dtype=np.float32)
    shape[:2] = mask.shape  # Since rfft2(img).shape != img.shape
    if len(shape) == 3:
        mask = mask[..., None]
    if abs(jitter_strength) > 1e-10:
        jitter = generate_uniform_noise(
            shape, 1 - jitter_strength, 1 + jitter_strength, dtype=np.float32
        )
        mask = jitter * mask
    # noise in frequency domain
    if noisef is None:
        noisef = (
            generate_gaussian_noise((*shape, 2), 0, 1, dtype=np.float32)
            .view(np.complex64)
            .squeeze(-1)
        )
    noisef = noisef * mask
    #
    noise = np.fft.irfft2(noisef, axes=(0, 1))
    if noise.shape[:2] != (h, w):
        noise = cv2.resize(noise, (w, h))
    return noise


#
def generate_pink_noise(
    shape: tuple[int, ...],
    noisef: np.ndarray | float | None = None,
    jitter_strength: float = 0,
):
    """Generate a 2D pink noise.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of noise, must be `(H, W)` or `(H, W, C)`
    noisef : np.ndarray | float | None, default=None
        The noise in frequency domain (by using `fft.rfft2`). If `None`
        is provided, the Gaussian noise is applied.
    jitter_strength: float, default=0
        Jitter the mask. Add extra uniform noise `[1-strength, 1+strength]`.

    Returns
    -------
    np.ndarray
        Pink noise with the given shape.
    """
    pink_noise = generate_power_noise(
        shape, noisef, beta=1, jitter_strength=jitter_strength
    )
    return pink_noise


def generate_red_noise(
    shape: tuple[int, ...],
    noisef: np.ndarray | float | None = None,
    jitter_strength: float = 0,
):
    """Generate a 2D red noise.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of noise, must be `(H, W)` or `(H, W, C)`
    noisef : np.ndarray | float | None, default=None
        The noise in frequency domain (by using `fft.rfft2`). If `None`
        is provided, the Gaussian noise is applied.
    jitter_strength: float, default=0
        Jitter the mask. Add extra uniform noise `[1-strength, 1+strength]`.

    Returns
    -------
    np.ndarray
        Red noise with the given shape.
    """
    red_noise = generate_power_noise(
        shape, noisef, beta=2, jitter_strength=jitter_strength
    )
    return red_noise


def generate_blue_noise(
    shape: tuple[int, ...],
    noisef: np.ndarray | float | None = None,
    jitter_strength: float = 0,
):
    """Generate a 2D blue noise.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of noise, must be `(H, W)` or `(H, W, C)`
    noisef : np.ndarray | float | None, default=None
        The noise in frequency domain (by using `fft.rfft2`). If `None`
        is provided, the Gaussian noise is applied.
    jitter_strength: float, default=0
        Jitter the mask. Add extra uniform noise `[1-strength, 1+strength]`.

    Returns
    -------
    np.ndarray
        Blue noise with the given shape.
    """
    blue_noise = generate_power_noise(
        shape, noisef, beta=-1, jitter_strength=jitter_strength
    )
    return blue_noise


def generate_purple_noise(
    shape: tuple[int, ...],
    noisef: np.ndarray | float | None = None,
    jitter_strength: float = 0,
):
    """Generate a 2D purple noise.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of noise, must be `(H, W)` or `(H, W, C)`
    noisef : np.ndarray | float | None, default=None
        The noise in frequency domain (by using `fft.rfft2`). If `None`
        is provided, the Gaussian noise is applied.
    jitter_strength: float, default=0
        Jitter the mask. Add extra uniform noise `[1-strength, 1+strength]`.

    Returns
    -------
    np.ndarray
        Purple noise with the given shape.
    """
    purple_noise = generate_power_noise(
        shape, noisef, beta=-2, jitter_strength=jitter_strength
    )
    return purple_noise


def fft_blend(img: np.ndarray, noise: np.ndarray, weight: float = 0.5):
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imgf = np.fft.rfft2(img, axes=(0, 1))
    noise_amp = np.abs(np.fft.rfft2(noise, axes=(0, 1)))
    # noise_amp = cv2.normalize(noise_amp, None, 0, 1, cv2.NORM_MINMAX)
    if noise_amp.ndim < imgf.ndim:
        noise_amp = noise_amp[..., None]
    amp = np.abs(imgf)
    scale = 1 + (noise_amp / amp) * weight
    new_imgf = scale * imgf
    res = np.fft.irfft2(new_imgf, axes=(0, 1))
    if res.shape[:2] != img.shape[:2]:
        res = cv2.resize(res, img.shape[:2][::-1])
    return res
