# This code is referenced from https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

import torch
import torch.nn.functional as F
from ..builder import LOSSES
import numpy as np

"""NOTE: Method1: use torch.tensor in GPU to compute PSNR and SSIM"""


def psnr(input_image, target_image):
    """Computes peak signal-to-noise ratio.input and target must be normalized to [0,1]"""
    return 10 * torch.log10(1 / (F.mse_loss(input_image, target_image) + 1e-6))  # if two is same, we get 6.


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _fspecial_gauss_2d(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    Args:
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
    Returns:
        torch.Tensor: 2D kernel (1 x 1 x size)
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    g = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    g = torch.from_numpy(g).to(dtype=torch.float)
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
         learnable parameters? this need gradient to train.
    Args:
        input (torch.Tensor): a batch of tensors to be blured 用来做高斯模糊的吗？
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    # 此时 win 就是卷积核对应的权重 依次为 N种C通道的H,W卷积核
    win = win.expand(C, win.size(1), win.size(2), win.size(3))
    out = F.conv2d(input, win, stride=1, padding=2, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=2, groups=C)
    return out


def _ssim(X, Y,
          data_range,
          win,
          size_average=True,
          K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y,
         data_range=255,
         size_average=True,
         win_size=11,
         win_sigma=1.5,
         win=None,
         K=(0.01, 0.03),
         nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    ssim_per_channel, cs = _ssim(X, Y,
                                 data_range=data_range,
                                 win=win,
                                 size_average=False,
                                 K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(X, Y,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            win=None,
            weights=None,
            K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                                     win=win,
                                     data_range=data_range,
                                     size_average=False,
                                     K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


def direction_aware_ssim(X,
                         Y,
                         data_range=255,
                         size_average=True,
                         win_size=11,
                         win_sigma=1.5,
                         win=None,
                         weights=None,
                         K=(0.01, 0.03)):
    r""" interface of direction_aware_ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                                     win=win,
                                     data_range=data_range,
                                     size_average=False,
                                     K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


@LOSSES.register_module()
class SSIM(torch.nn.Module):
    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=1,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y,
                    data_range=self.data_range,
                    size_average=self.size_average,
                    win=self.win,
                    K=self.K,
                    nonnegative_ssim=self.nonnegative_ssim)


@LOSSES.register_module()
class MS_SSIM(torch.nn.Module):
    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=1,
                 weights=None,
                 K=(0.01, 0.03)):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X, Y,
                       data_range=self.data_range,
                       size_average=self.size_average,
                       win=self.win,
                       weights=self.weights,
                       K=self.K)


@LOSSES.register_module()
class DirectionAwareSSIM(torch.nn.Module):
    def __init__(self,
                 data_range=255,  # 如此看来，需要恢复0-255颜色信息
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=1,
                 weights=None,
                 K=(0.01, 0.03)):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(DirectionAwareSSIM, self).__init__()
        self.win_size = win_size
        # self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.win = _fspecial_gauss_2d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return direction_aware_ssim(X, Y,
                                    data_range=self.data_range,
                                    size_average=self.size_average,
                                    win=self.win,
                                    weights=self.weights,
                                    K=self.K)


@LOSSES.register_module()
class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


@LOSSES.register_module()
class MSSSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MSSSIM_Loss, self).forward(img1, img2)


@LOSSES.register_module()
class DirectionAwareSSIM_Loss(DirectionAwareSSIM):
    def forward(self, img1, img2):
        return 1 - super(DirectionAwareSSIM_Loss, self).forward(img1, img2)
