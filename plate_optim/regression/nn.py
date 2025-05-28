import torch
import torch.nn as nn
import torch.nn.functional as F


class Film(nn.Module):
    """
    OWN IMPLEMENTATION OF FiLM LAYER
    A PyTorch implementation of the FiLM (Feature-wise Linear Modulation) layer.

    This layer applies feature-wise affine transformations to the input tensor `x`,
    where the scaling (weight) and shifting (bias) parameters are conditioned on 
    an additional input `conditional`. The affine transformations are applied 
    independently for each feature channel.

    Args:
        conditional_dim (int): The dimensionality of the conditional input.
        projection_dim (int): The dimensionality of the projected weight and bias outputs.
        **kwargs: Additional arguments passed to the `nn.Module` base class.

    Attributes:
        weight (nn.Sequential): A feedforward network generating the scaling (weight)
            parameters based on the `conditional` input.
        bias (nn.Sequential): A feedforward network generating the shifting (bias)
            parameters based on the `conditional` input.

    Methods:
        forward(x, conditional):
            Applies the FiLM transformation to the input `x`.

            Args:
                x (torch.Tensor): The input tensor with shape (batch_size, channels, ...),
                    where `...` represents additional spatial dimensions.
                conditional (torch.Tensor): The conditioning input tensor with shape
                    (batch_size, conditional_dim).

            Returns:
                torch.Tensor: The transformed tensor with the same shape as `x`.
    """
    def __init__(self, conditional_dim, projection_dim, **kwargs):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(conditional_dim, projection_dim, bias=False),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.bias = nn.Sequential(
            nn.Linear(conditional_dim, projection_dim, bias=False),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x, conditional):
        """
        Forward pass of the FiLM layer.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, channels, ...).
            conditional (torch.Tensor): The conditioning input tensor with shape
                (batch_size, conditional_dim).

        Returns:
            torch.Tensor: The transformed tensor with the same shape as `x`.
        """
        if len(conditional.shape) == 1:
            conditional = conditional.unsqueeze(1)
        ndim = len(x.shape) - 2
        view_shape = x.shape[:2] + (1,) * ndim
        return self.weight(conditional).view(*view_shape) * x + self.bias(conditional).view(*view_shape)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, add_time_embedding=False, add_extra=False, n_extra=0):
        super().__init__()
        self.add_time_embedding, self.add_extra = add_time_embedding, add_extra
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        if add_time_embedding:
            self.time_embedding = Film(1, out_channels)
        if add_extra:    
            self.extra_embedding = Film(n_extra, out_channels)

    def forward(self, x, t=None, extra=None):
        x = self.maxpool_conv(x)
        if self.add_time_embedding:
            x = self.time_embedding(x, t)
        if self.add_extra and extra is not None:
            x = self.extra_embedding(x, extra)        
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, add_time_embedding=False, add_extra=False, n_extra=0):
        super().__init__()
        self.add_time_embedding, self.add_extra = add_time_embedding, add_extra
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        if add_time_embedding:
            self.time_embedding = Film(1, out_channels)
        if add_extra:    
            self.extra_embedding = Film(n_extra, out_channels)

    def forward(self, x, skip_x, t=None, extra=None):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if self.add_time_embedding:
            x = self.time_embedding(x, t)
        if self.add_extra and extra is not None:
            x = self.extra_embedding(x, extra)    
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, *size)
