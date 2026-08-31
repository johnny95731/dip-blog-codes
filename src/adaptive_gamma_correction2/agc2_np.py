import cv2
import numpy as np


def agcwd(rgb: np.ndarray, alpha: float = 1.5):
    """An implementation of the adaptive gamma correction with weighting
    distribution (AGCWD).

    Parameters
    ----------
    img : np.ndarray
        An RGB or grayscale image with shape `(H, W, *)`.
    alpha : float, default=1.5
        A parameter for correcting the weights.

    References
    ----------
    [1] S. -C. Huang, F. -C. Cheng and Y. -S. Chiu, "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, March 2013, doi: 10.1109/TIP.2012.2226047
    """
    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        rgb = cv2.convertScaleAbs(
            cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
        )

    if num_ch == 3:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        gray = yuv[:, :, 0]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    pdf = cv2.calcHist([gray], [0], None, [256], [0, 255])
    pdf01 = cv2.normalize(pdf, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pdf_w = cv2.pow(pdf01, alpha)[:, 0]
    cdf_w = np.cumsum(pdf_w, -1)
    cdf_w = cdf_w / cdf_w[..., -1:]
    #
    gamma = 1 - cdf_w
    table = np.linspace(0, 1, 256, dtype=np.float32)
    table = np.pow(table, gamma)

    table = cv2.convertScaleAbs(table, None, 255)[:, 0]
    res = cv2.LUT(gray, table)
    if num_ch == 3:
        yuv[:, :, 0] = res
        res = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return res


def cagc(
    rgb: np.ndarray,
    bri_thresh: float = 0.5,
    tau: float = 3.0,
):
    """An implementation of the conditional adaptive gamma correction [1].

    Parameters
    ----------
    img : np.ndarray
        An RGB or grayscale image with shape `(H, W, *)`.
    bri_thresh : float, default=0.5
        The threshold of the brightness.
    tau : float, default=3.0
        The threshold of the contrast.

    References
    ----------
    [1] Rahman, S., Rahman, M.M., Abdullah-Al-Wadud, M. et al. An adaptive gamma correction for image enhancement. J Image Video Proc. 2016, 35 (2016). https://doi.org/10.1186/s13640-016-0138-1
    """
    num_ch = 1 if rgb.ndim == 2 else rgb.shape[2]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        rgb = cv2.convertScaleAbs(
            cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
        )

    if num_ch == 3:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        gray = yuv[:, :, 0]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')

    mean, std = cv2.meanStdDev(gray)
    mean = (mean.flatten() / 255).astype(np.float32)
    std = (std.flatten() / 255).astype(np.float32)

    is_dark = (mean < bri_thresh).item()
    is_low_contrast = (std < 1 / (4 * tau)).item()
    #
    x = np.linspace(0, 1, 256, dtype=np.float32)
    gamma = (-np.log2(std)) if is_low_contrast else np.exp((1 - mean - std) / 2)
    table = np.pow(x, gamma, out=x)
    if is_dark:
        mean **= gamma
        k = table + (1 - table) * mean
        table /= k

    table = cv2.convertScaleAbs(table, None, 255).squeeze(1)
    res = cv2.LUT(gray, table)
    if num_ch == 3:
        yuv[:, :, 0] = res
        res = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return res
