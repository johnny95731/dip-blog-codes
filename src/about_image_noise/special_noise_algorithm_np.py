import math

import cv2
import numpy as np

from .noise_np import generate_gaussian_noise


def voss_mccartney_2d(
    shape: tuple[int, ...] = (224, 224),
    num_octaves: int = 20,
    scale: float = 2,
    dtype: np.floating = np.float32,
):
    """Generate pink noise by Voss-McCartney algorithm.

    The code is generated

    Parameters
    ----------
    shape : tuple[int,...], default=(224, 224)
        The shape of noise map. Must be `(height, width)`
        or `(height, width, channel)`.
    num_octaves : int, default=10
        The number of octaves/loops.
    scale
    """
    assert 2 <= len(shape) <= 3, (
        '`shape` must be `(height, width)` or `(height, width, channel)`.'
    )
    assert np.issubdtype(dtype, np.floating), '`dtype` must be a floating type.'
    shape = list(shape)
    noise = generate_gaussian_noise(shape, dtype=dtype)
    ori_wh = shape[:2][::-1]  # (width, height) for cv2.resize
    for _ in range(1, num_octaves):
        # Add low resolution noise => increase low frequency
        h_down = math.ceil(shape[0] / scale)
        w_down = math.ceil(shape[1] / scale)
        shape[:2] = (h_down, w_down)
        low_res = generate_gaussian_noise(shape, dtype=dtype)
        high_res = cv2.resize(low_res, ori_wh, interpolation=cv2.INTER_CUBIC)

        noise += high_res
    noise -= np.mean(noise, (0, 1))
    noise /= np.std(noise, (0, 1))
    noise = cv2.GaussianBlur(noise, (7, 7), 0)
    return noise


def generate_perlin_noise_2d(shape, res):
    """
    使用純 NumPy 向量化產生 2D Perlin 雜訊
    - shape: 產生的影像尺寸 (height, width)，必須是 res 的倍數
    - res: 網格頻率 (res_y, res_x)
    """

    def f(t):
        # 5次多項式平滑函數 (Smoothstep)
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # 建立小數座標網格
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(
            1, 2, 0
        )
        % 1
    )

    # 隨機梯度方向 (Angles & Gradients)
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    np.repeat()
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # 計算各個角落的梯度點積 (Ramps)
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum((grid - np.array([1, 0])) * g10, axis=2)
    n01 = np.sum((grid - np.array([0, 1])) * g01, axis=2)
    n11 = np.sum((grid - np.array([1, 1])) * g11, axis=2)

    # Bi-linear interpolarison
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + n10 * t[:, :, 0]
    n1 = n01 * (1 - t[:, :, 0]) + n11 * t[:, :, 0]
    noise = n0 * (1 - t[:, :, 1]) + n1 * t[:, :, 1]

    return noise


def generate_fractal_noise_2d(shape, res, octaves=4, persistence=0.5):
    """
    疊加多個八音階 (Octaves) 產生豐富的碎形 Perlin 雜訊
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1])
        )
        frequency *= 2
        amplitude *= persistence
    return noise
