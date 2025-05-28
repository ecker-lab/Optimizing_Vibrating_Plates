import torch
import torch.nn as nn
from torchinterp1d import interp1d
from plate_optim.regression.nn import DoubleConv, Down, Up, SelfAttention, Film


class convert_1d_to_interpolator:
    def __init__(self, array, min_val, max_val):
        self.array = array
        self.min_val = min_val
        self.max_val = max_val
        self.x = torch.linspace(min_val, max_val, steps=array.shape[0], device=array.device)

    def __getitem__(self, xnew):
        if not isinstance(xnew, torch.Tensor):
            xnew = torch.tensor(xnew, dtype=torch.float32, device=self.array.device)
        original_shape = xnew.shape
        xnew_flat = xnew.flatten()
        interpolated_values_flat = interp1d(self.x, self.array, xnew_flat, None)
        return interpolated_values_flat.view(original_shape)

    def cuda(self):
        self.array = self.array.cuda()
        self.x = self.x.cuda()
        return self

    def to(self, device, dtype=None):
        self.array = self.array.to(device, dtype=dtype)
        self.x = self.x.to(device, dtype=dtype)
        return self


def get_mean_from_velocity_field(field_solution, field_mean, field_std, frequencies=None):
    v_ref = 1e-9
    eps = 1e-12
    B, n_frequencies = field_solution.shape[:2]
    # remove transformations in dataloader
    if frequencies is None:
        field_solution = field_solution * field_std + field_mean
    else:
        field_solution = field_solution * field_std + field_mean[frequencies].unsqueeze(-1).unsqueeze(-1)
    field_solution = torch.exp(field_solution) - eps
    # calculate frequency response
    v = torch.mean(field_solution.view(B, n_frequencies, -1), dim=-1)
    v = 10 * torch.log10((v + 1e-12) / v_ref)
    return v.view(B, n_frequencies)


class FQO_Unet(nn.Module):
    def __init__(self, c_in=1, conditional=False, scaling_factor=32, len_conditional=None, **kwargs):
        super().__init__()
        k = scaling_factor
        self.inc = DoubleConv(c_in, 2 * k)
        self.down0 = nn.Sequential(nn.Conv2d(2 * k, 2 * k, 3, stride=2, padding=1), nn.ReLU())

        self.down1 = Down(2 * k, 4 * k)
        self.down2 = Down(4 * k, 6 * k)
        self.sa2 = SelfAttention(6 * k)
        self.down3 = Down(6 * k, 8 * k)
        self.sa3 = SelfAttention(8 * k)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(len_conditional, 6 * k)
        self.bot1 = DoubleConv(8 * k, 8 * k)
        self.bot3 = DoubleConv(8 * k, 6 * k)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_film = Film(6 * k, 6 * k)
        self.queryfilm1 = Film(1, 6 * k)

        self.up_project1 = DoubleConv(6 * k, 2 * k)
        self.up1 = Up(8 * k, 4 * k)
        self.sa4 = SelfAttention(4 * k)
        self.queryfilm2 = Film(1, 4 * k)

        self.up_project2 = DoubleConv(4 * k, 2 * k)
        self.up2 = Up(6 * k, 3 * k)
        self.queryfilm3 = Film(1, 3 * k)

        self.up_project3 = DoubleConv(2 * k, 1 * k)
        self.up3 = Up(4 * k, 2 * k)
        self.outc = nn.Conv2d(2 * k, 1, kernel_size=1)
        self.nfe_counter = 0

    def reset_nfe_counter(self):
        self.nfe_counter = 0

    def forward_encoder(self, x, conditional):
        self.nfe_counter += x.shape[0]
        x = torch.nn.functional.interpolate(x, size=(96, 128), mode='bilinear', align_corners=True)
        x = self.inc(x)
        x1 = self.down0(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x3 = self.sa2(x3)
        if self.conditional is True:
            x3 = self.film(x3, conditional)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot3(x4)
        gap = self.global_avg_pool(x4)[:, :, 0, 0]
        x4 = self.gap_film(x4, gap)
        return x1, x2, x3, x4

    def forward(self, x, conditional, frequencies):
        B, n_freqs = frequencies.shape
        x1, x2, x3, x4 = self.forward_encoder(x, conditional)

        x4 = x4.repeat_interleave(n_freqs, dim=0)
        x4 = self.queryfilm1(x4, frequencies.view(-1, 1))
        x = self.up1(x4, self.up_project1(x3).repeat_interleave(n_freqs, dim=0))
        x = self.sa4(x)
        x = self.queryfilm2(x, frequencies.view(-1, 1))
        x = self.up2(x, self.up_project2(x2).repeat_interleave(n_freqs, dim=0))
        x = self.queryfilm3(x, frequencies.view(-1, 1))
        x = self.up3(x, self.up_project3(x1).repeat_interleave(n_freqs, dim=0))

        output = self.outc(x)
        output = torch.nn.functional.interpolate(output, size=(61, 91), mode='bilinear', align_corners=True)
        return output.reshape(B, n_freqs, output.size(2), output.size(3))


def get_net(conditional=False, len_conditional=None, scaling_factor=32):
    return FQO_Unet(conditional=conditional, len_conditional=len_conditional, scaling_factor=scaling_factor)
