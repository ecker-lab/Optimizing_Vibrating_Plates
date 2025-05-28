import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn


def softlocalmax(function: torch.tensor, frequencies: torch.tensor = None, alpha=1.0):
    """
    Args:
        function: shape  (N,)
        frequencies: shape (N,)
        alpha: 
            The parameter makes the location of the peaks more precise,
            at the cost of the quality of the gradients

    Returns:
        A list of peak positions
    """

    if frequencies is None:
        frequencies = torch.arange(1, function.shape[0] + 1, dtype=function.dtype, device=function.device)

    function_pad = np.pad(function.detach().cpu().numpy(), pad_width=2, mode="symmetric")

    peak_positions = find_peaks(function_pad)[0]

    peak_positions = np.clip(peak_positions - 2, 0, len(function))

    peak_intervall_dividers = (peak_positions[1:] + peak_positions[:-1]) / 2

    peak_intervall_dividers = np.concatenate([[0], peak_intervall_dividers, [len(function) - 1]])

    softargmaxes = []
    for start, end in zip(peak_intervall_dividers[:-1], peak_intervall_dividers[1:]):
        x = function[int(start):int(end)]
        softargmax = (torch.softmax(x * alpha, dim=0) @ frequencies[int(start):int(end)])

        softargmaxes.append(softargmax)

    return softargmaxes


def create_gaussian_kernel(sigma):
    l = np.round(3 * sigma + 1.0)
    x = np.linspace(-(l - 1.0) / 2.0, (l - 1.0) / 2.0, int(l))
    kernel = np.exp(-0.5 * np.square(x) / np.square(sigma))
    return kernel / np.sum(kernel)


def position_loss(pred_fr_funcs, interval, freqs):
    """
    calculates from fr_funcs, the loss, that punishes 
    peaks, that lie inside the interval

    pred_fr_funcs: torch.tensor shape (B,F)
        the frequency response functions, 
        it is not imported how they are normalized, 
        as long as scipy's find_peaks works on them
    
    interval: tuple of (start,end)
    freqs: torch.tensor shape (F,)
        the frequencies for the pred_fr_funcs

    returns: torch.tensor of shape (B,)
        The tensor contains the loss per frequency response function
    """
    device = pred_fr_funcs.device
    dtype = pred_fr_funcs.dtype
    losses = []
    for pred_fr_func in pred_fr_funcs:
        peaks = softlocalmax(pred_fr_func, freqs)
        peaks = torch.stack(peaks)
        mask = (peaks > interval[0]) & (peaks < interval[1])

        peaks_in_interval = peaks[mask]
        mid_point = (interval[0] + interval[1]) / 2.0
        # find the peaks that are the closest to the mid point
        if mask.any() == False:
            sorted_diffs = torch.sort(-((peaks - mid_point)**2))
            losses.append(sorted_diffs.values[-2:].sum())
            #losses.append(sorted_diffs.values[0:0].sum())
            # losses.append(torch.tensor(torch.nan,device=pred_fr_funcs.device,dtype=pred_fr_funcs.dtype))
        else:
            losses.append((-((peaks_in_interval - mid_point)**2)).sum())

    return torch.stack(losses)


class Gaussian1DBlur(nn.Module):
    """
    This class smooths the frequency response functions with an 
    gaussian blur 
    """

    def __init__(self, sigma=2):
        super().__init__()
        kernel = create_gaussian_kernel(sigma)
        kernel = torch.from_numpy(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()

        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, fr_funcs):
        x_in = fr_funcs.unsqueeze(1)
        tr_blured = torch.nn.functional.conv1d(x_in, self.kernel.to(x_in.device), padding="same")
        return tr_blured[:, 0]


class PositionIntervalLoss(nn.Module):
    """
    This loss should move the peaks outside of an interval

    Example for using this loss function:
    ------------------------------------------------------
    loss_fn = PositionIntervalLoss((100,200),sigma=2)

    #we have to move the loss fn to the device
    loss_fn = loss_fn.to(device)

    fr_funcs = fr_funcs.to(device)
    losses = loss_fn(fr_funcs)
    ...
    -----------------------------------------------------
    """
    def __init__(self, interval, freqs=None, sigma=0, scale_factor=1.0, device='cuda'):
        """
        interval: tuple of (start,end)
            the interval in which no peaks should lie
        freqs: torch.tensor of shape (F,)
            the frequencies of the fr functions.
            The freqs should not be normalized.
            Also it is recommended, that freqs.min()-20<interval[0]
            and freqs.max()+20>interval[1]
        sigma: float
            the stdev of the gaussian kernel, with which the fr functions will be 
            smoothed
        scale_factor: float
            With this factor the loss can be scaled (probably not needed anymore)
        """
        super().__init__()
        if freqs is None:
            freqs = torch.arange(1, 301).to(device)
        freqs = (freqs - interval[0]) / (interval[1] - interval[0])
        freqs = freqs * 2.0 - 1.0
        self.register_buffer("freqs", freqs, persistent=False)
        if sigma > 0:
            self.preprocessor = Gaussian1DBlur(sigma)
        else:
            self.preprocessor = nn.Identity()
        self.interval = (-1, 1)
        self.scale_factor = scale_factor

    def forward(self, fr_funcs):
        """
        fr_funcs: tensor of shape (B,F)

        returns:tensor of shape (B,)
            the loss per fr function.
        """
        fr_funcs = self.preprocessor(fr_funcs)
        return position_loss(fr_funcs, self.interval, self.freqs.to(fr_funcs.device)) * self.scale_factor


class PeakMoverLoss(nn.Module):
    """
    This loss tries to move the first peak to the right 
    """

    def __init__(self,freqs=None, sigma=0, device='cuda'):
        """
        freqs: torch.tensor of shape (F,)
            the frequencies of the fr functions.
            The freqs should not be normalized.
            If we want the first peak of the fr func,
            the freqs should start at 1 

        sigma: float
            the stdev of the gaussian kernel, with which the fr functions will be 
            smoothed
        """
        super().__init__()
        if freqs is None:
            freqs = torch.arange(1,301).to(device)

        #scale freqs between 0 and 1 
        freqs = (freqs-freqs.min())/(freqs.max()-freqs.min())
        #scale freqs between -1 and 1
        freqs = freqs*2.-1.
        
        self.register_buffer("freqs", freqs, persistent=False)

        if sigma > 0:
            self.preprocessor = Gaussian1DBlur(sigma)
        else:
            self.preprocessor = nn.Identity()

    def forward(self, fr_funcs):
        """
        fr_funcs: tensor of shape (B,F)

        returns:tensor of shape (B,)
            the loss per fr function.
        """
        fr_funcs = self.preprocessor(fr_funcs)
        losses = []
        for fr_func in fr_funcs:
            peaks = softlocalmax(fr_func, self.freqs.to(fr_func.device))
            peaks = torch.stack(peaks)
            losses.append(-peaks[0])
        return torch.stack(losses) 
