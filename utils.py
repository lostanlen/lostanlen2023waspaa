import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple
from speechbrain.processing.signal_processing import (
    gabor_impulse_response,
    gabor_impulse_response_legacy_complex,
)



class GaborConv1d(nn.Module):
    """
    This class implements 1D Gabor Convolutions from

    Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc. of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    padding : str
        (same, valid). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    sample_rate : int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_freq : float
        Lowest possible frequency (in Hz) for a filter
    max_freq : float
        Highest possible frequency (in Hz) for a filter
    n_fft: int
        number of FFT bins for initialization
    normalize_energy: bool
        whether to normalize energy at initialization. Default is False
    bias : bool
        If True, the additive bias b is adopted.
    sort_filters: bool
        whether to sort filters by center frequencies. Default is False
    use_legacy_complex: bool
        If False, torch.complex64 data type is used for gabor impulse responses
        If True, computation is performed on two real-valued tensors
    skip_transpose: bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 8000])
    >>> # 401 corresponds to a window of 25 ms at 16000 kHz
    >>> gabor_conv = GaborConv1d(
    ...     40, kernel_size=401, stride=1, in_channels=1
    ... )
    >>> #
    >>> out_tensor = gabor_conv(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 8000, 40])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride,
        input_shape=None,
        in_channels=None,
        padding="same",
        padding_mode="constant",
        sample_rate=16000,
        min_freq=60.0,
        max_freq=None,
        n_fft=512,
        normalize_energy=False,
        bias=False,
        sort_filters=False,
        use_legacy_complex=False,
        skip_transpose=False,
    ):
        super(GaborConv1d, self).__init__()
        self.filters = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.sort_filters = sort_filters
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        if max_freq is None:
            max_freq = sample_rate / 2
        self.max_freq = max_freq
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy
        self.use_legacy_complex = use_legacy_complex
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.kernel = nn.Parameter(self._initialize_kernel())
        if bias:
            self.bias = torch.nn.Parameter(torch.ones(self.filters * 2,))
        else:
            self.bias = None

    def forward(self, x):
        """Returns the output of the Gabor convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        kernel = self._gabor_constraint(self.kernel)
        if self.sort_filters:
            idxs = torch.argsort(kernel[:, 0])
            kernel = kernel[idxs, :]

        filters = self._gabor_filters(kernel)
        if not self.use_legacy_complex:
            temp = torch.view_as_real(filters)
            real_filters = temp[:, :, 0]
            img_filters = temp[:, :, 1]
        else:
            real_filters = filters[:, :, 0]
            img_filters = filters[:, :, 1]
        stacked_filters = torch.cat(
            [real_filters.unsqueeze(1), img_filters.unsqueeze(1)], dim=1
        )
        stacked_filters = torch.reshape(
            stacked_filters, (2 * self.filters, self.kernel_size)
        )
        stacked_filters = stacked_filters.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size)
        elif self.padding == "valid":
            pass
        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )

        output = F.conv1d(
            x, stacked_filters, bias=self.bias, stride=self.stride, padding=0
        )
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output

    def _gabor_constraint(self, kernel_data):
        mu_lower = 0.0
        mu_upper = math.pi
        sigma_lower = (
            4
            * torch.sqrt(
                2.0 * torch.log(torch.tensor(2.0, device=kernel_data.device))
            )
            / math.pi
        )
        sigma_upper = (
            self.kernel_size
            * torch.sqrt(
                2.0 * torch.log(torch.tensor(2.0, device=kernel_data.device))
            )
            / math.pi
        )
        clipped_mu = torch.clamp(
            kernel_data[:, 0], mu_lower, mu_upper
        ).unsqueeze(1)
        clipped_sigma = torch.clamp(
            kernel_data[:, 1], sigma_lower, sigma_upper
        ).unsqueeze(1)
        return torch.cat([clipped_mu, clipped_sigma], dim=-1)

    def _gabor_filters(self, kernel):
        t = torch.arange(
            -(self.kernel_size // 2),
            (self.kernel_size + 1) // 2,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        if not self.use_legacy_complex:
            return gabor_impulse_response(
                t, center=kernel[:, 0], fwhm=kernel[:, 1]
            )
        else:
            return gabor_impulse_response_legacy_complex(
                t, center=kernel[:, 0], fwhm=kernel[:, 1]
            )

    def _manage_padding(self, x, kernel_size):
        # this is the logic that gives correct shape that complies
        # with the original implementation at https://github.com/google-research/leaf-audio

        def get_padding_value(kernel_size):
            """Gets the number of elements to pad."""
            kernel_sizes = (kernel_size,)
            from functools import reduce
            from operator import __add__

            conv_padding = reduce(
                __add__,
                [
                    #(k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
                    (k // 2 + (k - 2 * (k // 2)), k // 2)
                    for k in kernel_sizes[::-1]
                ],
            )
            return conv_padding

        pad_value = get_padding_value(kernel_size)
        x = F.pad(x, pad_value, mode=self.padding_mode, value=0)
        return x

    def _mel_filters(self):
        def _mel_filters_areas(filters):
            peaks, _ = torch.max(filters, dim=1, keepdim=True)
            return (
                peaks
                * (torch.sum((filters > 0).float(), dim=1, keepdim=True) + 2)
                * np.pi
                / self.n_fft
            )

        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.filters,
            sample_rate=self.sample_rate,
        )
        mel_filters = mel_filters.transpose(1, 0)
        if self.normalize_energy:
            mel_filters = mel_filters / _mel_filters_areas(mel_filters)
        return mel_filters

    def _gabor_params_from_mels(self):
        coeff = torch.sqrt(2.0 * torch.log(torch.tensor(2.0))) * self.n_fft
        sqrt_filters = torch.sqrt(self._mel_filters())
        center_frequencies = torch.argmax(sqrt_filters, dim=1)
        peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
        half_magnitudes = peaks / 2.0
        fwhms = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=1)
        output = torch.cat(
            [
                (center_frequencies * 2 * np.pi / self.n_fft).unsqueeze(1),
                (coeff / (np.pi * fwhms)).unsqueeze(1),
            ],
            dim=-1,
        )
        return output

    def _initialize_kernel(self):
        return self._gabor_params_from_mels()

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = 1
        else:
            raise ValueError(
                "GaborConv1d expects 2d or 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels

