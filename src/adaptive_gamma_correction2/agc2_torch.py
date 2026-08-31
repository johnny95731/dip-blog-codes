from typing import Literal

import torch


def __default_dtype(x: torch.Tensor) -> torch.dtype:
    dtype = x.dtype if torch.is_floating_point(x) else torch.float32
    return dtype


def rgb_to_yuv(
    rgb: torch.Tensor,
    standard: Literal['bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'] = 'bt.601',
) -> torch.Tensor:
    """Converts an image from RGB space to YUV space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB imag in the range of [0, 1] with shape `(*, 3, H, W)`.
    standard : {'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'}, default='bt.601'
        The specification. The chrominance channels are normalized to the
        range [-0.5, 0.5].

    Returns
    -------
    torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`. The range of Y is [0, 1]
        and the range of U and V are [-0.5, 0.5].
    """
    # fmt: off
    dtype = __default_dtype(rgb)
    device = rgb.device
    # All chrominance channel are normalized to the range [-0.5, 0.5].
    if standard == 'bt.601':
        matrix = torch.tensor(
            [[0.299,  0.587,  0.114],
            [-0.169, -0.331,  0.500],
            [ 0.500, -0.419, -0.081]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.709':
        matrix = torch.tensor(
            [[0.2126,  0.7152,  0.0722],
            [-0.1146, -0.3854,  0.5000],
            [ 0.5000, -0.4542, -0.0458]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.2020':
        matrix = torch.tensor(
            [[0.2627,  0.6780,  0.0593],
            [-0.1396, -0.3604,  0.5000],
            [ 0.5000, -0.4598, -0.0402]],
            dtype=dtype,
            device=device
        )
    elif standard == 'yiq':
        matrix = torch.tensor(
            [[0.30,  0.59,   0.11],
            [ 0.5000, -0.2315, -0.2685],
            [ 0.2028, -0.5000, 0.2972]],
            dtype=dtype,
            device=device
        )
    elif standard == 'ycocg':
        matrix = torch.tensor(
            [[0.25, 0.5,  0.25],
            [ 0.50, 0.0, -0.50],
            [-0.25, 0.5, -0.25]],
            dtype=dtype,
            device=device
        )
    else:
        raise ValueError(f'Invalid value of argument `standard`: {standard}')
    # fmt: on
    yuv = torch.einsum('...oc,...chw->...ohw', matrix, rgb)
    return yuv


def yuv_to_rgb(
    yuv: torch.Tensor,
    standard: Literal['bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'] = 'bt.601',
) -> torch.Tensor:
    """Converts an image from YUV space to RGB space.

    The input is assumed to be in the range of [0, 1] (for Y channel) and
    [-0.5, 0.5] (for U and V channels). The output will be clip to [0, 1].

    Parameters
    ----------
    yuv : torch.Tensor
        An image in YUV space with shape `(*, 3, H, W)`.
    standard : {'bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'}, default='bt.601'
        The specification. The chrominance channels are normalized to the
        range [-0.5, 0.5].

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape `(*, 3, H, W)`.
    """
    dtype = __default_dtype(yuv)
    device = yuv.device
    # fmt: off
    if standard == 'bt.601':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.4020],
            [ 1.0, -0.3441, -0.7141],
            [ 1.0,  1.7720,  0.0000]],
            dtype=dtype,
            device=device
        )
    elif standard == 'bt.709':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.5748],
            [ 1.0, -0.1873, -0.4681],
            [ 1.0,  1.8556,  0.0000]],
            dtype=dtype,
            device=yuv.device
        )
    elif standard == 'bt.2020':
        inv_matrix = torch.tensor(
            [[1.0,  0.0000,  1.4746],
            [ 1.0, -0.1646, -0.5714],
            [ 1.0,  1.8814,  0.0000]],
            dtype=dtype,
            device=device
        )
    elif standard == 'yiq':
        inv_matrix = torch.tensor(
            [[1.0,  1.1344,  0.6548],
            [ 1.0, -0.3292, -0.6676],
            [ 1.0, -1.3280,  1.7949]],
            dtype=dtype,
            device=device
        )
    elif standard == 'ycocg':
        inv_matrix = torch.tensor(
            [[1.0,  1.0, -1.0],
            [ 1.0,  0.0,  1.0],
            [ 1.0, -1.0, -1.0]],
            dtype=dtype,
            device=device
        )
    else:
        raise ValueError(f'Invalid value of argument `standard`: {standard}')
    # fmt: on
    rgb = torch.einsum('...oc,...chw->...ohw', inv_matrix, yuv).clip_(0.0, 1.0)
    return rgb


def histogram(
    img: torch.Tensor,
    bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the histogram of an image.

    Parameters
    ----------
    img : torch.Tensor
        An image in the range of [0, 1] with shape `(*, H, W)`
    bins : int, default=256
        The number of groups in data range.

    Returns
    -------
    hist : torch.Tensor
        The histogram or density. Shape `(*, bins)`.
    img_idx : torch.Tensor
        The index of histogram for each pixel. Shape `(*, H * W)`.
        The `img_idx` is returned only if `ret_index` is true.

    Examples
    --------

    >>> from imgtools.statistics import histogram
    >>>
    >>> img = torch.rand(3, 512, 512)
    >>> _hist1 = histogram(img)  # torch.Size([3, 256])
    >>> _hist2 = histogram(img, bins=300)  # torch.Size([3, 300])
    >>> _hist3 = mean(img, density=True)  # torch.Size([3, 256])
    >>> _hist3.sum(-1)  # (1., 1., 1.)
    """
    if not isinstance(bins, int):
        raise TypeError(f'`bins` must be an integer: {type(bins)}.')
    idx = (img * (bins - 1)).round_().long().clip_(0, bins - 1)

    flatted_idx = idx.flatten(start_dim=-2)
    hist = torch.zeros(
        img.shape[:-2] + (bins,),
        dtype=torch.float32,
        device=img.device,
    )
    hist.scatter_add_(
        dim=-1, index=flatted_idx, src=hist.new_ones(1).expand_as(flatted_idx)
    )
    num_el = flatted_idx.size(-1)
    hist = hist / num_el
    return hist, idx


def agcwd(rgb: torch.Tensor, alpha: float = 1.5, bins: int = 256):
    """An implementation of the adaptive gamma correction with weighting
    distribution (AGCWD).

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image in the range of `[0, 1]` with
        shape `(*, C, H, W)`.
    alpha : float, default=1.5
        A parameter for correcting the weights.
    bins : int, default=256
        The number of groups in data range.

    References
    ----------
    [1] S. -C. Huang, F. -C. Cheng and Y. -S. Chiu, "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, March 2013, doi: 10.1109/TIP.2012.2226047
    """
    dtype = __default_dtype(rgb)
    device = rgb.device
    num_ch = rgb.size(-3)
    if num_ch == 3:
        yuv = rgb_to_yuv(rgb)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    pdf, idx = histogram(gray, bins)
    mini_pdf = pdf.amin(-1, keepdim=True)
    maxi_pdf = pdf.amax(-1, keepdim=True)
    pdf_w = (pdf.sub_(mini_pdf)).div_(maxi_pdf - mini_pdf).pow_(alpha)
    cdf_w = torch.cumsum(pdf_w, -1, dtype=dtype)
    cdf_w = cdf_w / cdf_w[..., -1:]
    #
    gamma = 1 - cdf_w
    table = (
        torch
        .linspace(0, 1, bins, dtype=dtype, device=device)
        .expand_as(gamma)
        .pow_(gamma)
    )

    flatted_idx = idx.flatten(-2)
    res = torch.gather(table, -1, index=flatted_idx).reshape(gray.shape)
    if num_ch == 3:
        yuv[..., :1, :, :] = res
        res = yuv_to_rgb(yuv)
    return res


def cagc(
    rgb: torch.Tensor,
    bri_thresh: float = 0.5,
    tau: float = 3.0,
    bins: int = 256,
    yuv_std: Literal['bt.601', 'bt.709', 'bt.2020', 'yiq', 'ycocg'] = 'bt.601',
):
    """An implementation of the conditional adaptive gamma correction [1].

    Parameters
    ----------
    img : torch.Tensor
        An RGB or grayscale image in the range of `[0, 1]` with
        shape `(*, C, H, W)`.
    bri_thresh : float, default=0.5
        The threshold of the brightness.
    tau : float, default=3.0
        The threshold of the contrast.
    bins : int, default=256
        The number of groups in data range.
    yuv_std : {"bt.601", "bt.709", "bt.2020", "yiq", "ycocg"}, default="bt.601"
        The standard of YUV space.

    References
    ----------
    [1] Rahman, S., Rahman, M.M., Abdullah-Al-Wadud, M. et al. An adaptive gamma correction for image enhancement. J Image Video Proc. 2016, 35 (2016). https://doi.org/10.1186/s13640-016-0138-1
    """
    dtype = __default_dtype(rgb)
    device = rgb.device
    num_ch = rgb.size(-3)
    if num_ch == 3:
        yuv = rgb_to_yuv(rgb, yuv_std)
        gray = yuv[..., :1, :, :]
    elif num_ch == 1:
        gray = rgb
    else:
        raise ValueError(f'`rgb` must be 1 or 3 channel: {num_ch}')
    n = gray.shape[-1] * gray.shape[-2]
    pdf, idx = histogram(gray, bins)
    x = torch.linspace(0, 1, bins, dtype=dtype, device=device).expand_as(pdf)
    mean = (x * pdf).sum(-1, keepdim=True)
    std = (
        (x - mean)
        .square_()
        .mul_(pdf)
        .sum(-1, keepdim=True)
        .mul_(n / (n - 1))  # unbiased
        .sqrt_()
    )

    is_dark = mean < bri_thresh
    is_low_contrast = std < 1 / (4 * tau)
    #
    gamma = torch.where(
        is_low_contrast,
        std.log2().neg_(),
        torch.sub(0.5, mean + std, alpha=0.5).exp_(),
    )
    table = x.pow_(gamma)
    k = (1 - table).mul_(mean.pow_(gamma)).add_(table)
    c0 = torch.div(table, k, out=k)
    table = torch.where(is_dark, c0, table)

    flatted_idx = idx.flatten(-2)
    res = torch.gather(table, -1, index=flatted_idx).reshape(gray.shape)
    if num_ch == 3:
        yuv[..., :1, :, :] = res
        res = yuv_to_rgb(yuv, yuv_std)
    return res
