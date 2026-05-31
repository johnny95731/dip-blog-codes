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

import cv2
import numpy as np


def _laplacian_matrix(row: int, col: int):
    kx = np.fft.rfftfreq(col).astype(np.float32)
    kx *= 2 * np.pi
    np.cos(kx, out=kx)
    np.multiply(kx, -2, out=kx)
    kx = kx.reshape(1, -1)

    ky = np.fft.fftfreq(row).astype(np.float32)
    ky *= 2 * np.pi
    np.cos(ky, out=ky)
    np.multiply(ky, -2, out=ky)
    ky = ky.reshape(-1, 1)

    temp = kx + ky
    fft_laplacian = np.add(temp, 4.0, out=temp)
    return fft_laplacian


def _weighting_strategy(t_hat: np.ndarray, strategy: int):
    if strategy == 2:
        mat_w = np.stack(_derivative(t_hat), 0)
        np.abs(mat_w, out=mat_w)
        np.add(mat_w, 1.0, out=mat_w)
        np.divide(1.0, mat_w, out=mat_w)
        return mat_w
    else:
        h, w = t_hat.shape
        return np.ones((2, h, w))


def _derivative(
    t_hat: np.ndarray,
    direction: Literal['y', 'x', 'both'] = 'both',
):
    kernel = np.array(((-1, 1),), dtype=np.float32)
    dy = None
    dx = None
    if direction == 'y' or direction == 'both':
        dy = cv2.filter2D(
            t_hat, cv2.CV_32F, np.ascontiguousarray(kernel.T), anchor=(0, 0)
        )
    if direction == 'x' or direction == 'both':
        dx = cv2.filter2D(t_hat, cv2.CV_32F, kernel, anchor=(0, 0))
    return dy, dx


def _subproblem_t(
    t_hat: np.ndarray,
    img_t: np.ndarray,
    img_g: np.ndarray,
    img_z: np.ndarray,
    mu: float,
    # const
    fft_laplacian: np.ndarray,
    two: np.ndarray,
    # preallocate
    temp: np.ndarray,
    _numerator: np.ndarray,
    _denominator: np.ndarray,
):
    x = cv2.scaleAdd(img_z, -1 / mu, img_g, dst=temp)
    dy = x[0]
    dx = x[1]
    dyx = _derivative(dy, 'x')[1]
    dxy = _derivative(dx, 'y')[0]

    cv2.addWeighted(dyx, mu, dxy, mu, 0.0, dst=_numerator)
    cv2.scaleAdd(t_hat, 2.0, _numerator, dst=_numerator)
    numerator = np.fft.rfft2(_numerator)
    denominator = cv2.scaleAdd(fft_laplacian, mu, two, dst=_denominator)
    numerator /= denominator
    img_t = np.fft.irfft2(numerator, img_t.shape, out=img_t)
    return img_t


def _subproblem_g(
    dt: np.ndarray,
    img_g: np.ndarray,
    img_z: np.ndarray,
    mu: float,
    mat_w: np.ndarray,
    alpha: float,
    tempw: np.ndarray,
):
    epsilon = np.multiply(alpha / mu, mat_w, out=tempw)
    x = cv2.scaleAdd(img_z, 1 / mu, dt, dst=img_g)
    # sign(X) * max(abs(X) - epsilon, 0)
    sign = np.sign(x)
    abs_x = np.abs(x, out=x)
    np.subtract(abs_x, epsilon, out=img_g)
    np.clip(img_g, 0.0, None, out=img_g)
    np.multiply(sign, img_g, out=img_g)
    return img_g


def _subproblem_z(
    dt: np.ndarray,
    mat_g: np.ndarray,
    mat_z: np.ndarray,
    mu: float,
):
    mat_z += cv2.addWeighted(dt, mu, mat_g, -mu, 0.0, dst=dt)
    return mat_z


def apply_lime(
    img: np.ndarray,
    num_iter: int = 5,
    alpha: float = 2,
    rho: float = 2,
    gamma: float = 0.5,
    strategy=2,
):
    """An implements of Guo's work [1]. The papper is availible on arxiv [2].

    Parameters
    ----------
    img : np.ndarray
        Image in the range of [0, 1] with shape (H, W, C).
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
    np.ndarray
        Enhanced image. shape (H, W, C).

    References
    ----------
    [1] Guo X, Li Y, Ling H. LIME: Low-Light Image Enhancement via
        Illumination Map Estimation. IEEE Transactions on Image Processing
        2017, 26 (2), 982-993. https://doi.org/10.1109/TIP.2016.2639450.
    [2] https://arxiv.org/abs/1605.05034
    """
    h, w, _ = img.shape
    if isinstance(img.dtype, np.uint8):
        img = np.divide(img, 255, dtype=np.float32)
    # constants
    t_hat = np.max(img, -1)
    fft_laplacian = _laplacian_matrix(h, w)
    mat_w = _weighting_strategy(t_hat, strategy)
    # iteration variables
    img_t = np.zeros((h, w), np.float32)
    img_g = np.zeros((2, h, w), np.float32)
    img_z = np.zeros((2, h, w), np.float32)
    mu = 1.0
    # temp variables
    tempt = np.empty_like(img_t)
    tempg = np.empty_like(img_g)
    templap = np.empty_like(fft_laplacian)
    two = np.full_like(fft_laplacian, 2.0)

    img_t = _subproblem_t(
        t_hat,
        img_t,
        img_g,
        img_z,
        mu,
        fft_laplacian,
        two,
        tempg,
        tempt,
        templap,
    )
    for _ in range(num_iter - 1):
        dt = np.stack(_derivative(img_t), 0)
        _subproblem_g(dt, img_g, img_z, mu, mat_w, alpha, tempg)
        _subproblem_z(dt, img_g, img_z, mu)
        mu *= rho
        img_t = _subproblem_t(
            t_hat,
            img_t,
            img_g,
            img_z,
            mu,
            fft_laplacian,
            two,
            tempg,
            tempt,
            templap,
        )
    np.clip(img_t, 0.0, None, out=img_t)
    img_t **= gamma
    img_t = img_t[..., None]

    res = np.divide(img, img_t, out=img.copy(), where=img_t > 0)
    np.clip(res, 0.0, 1.0, out=res)
    return res
