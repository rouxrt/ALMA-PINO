import torch
import torch.nn as nn

from models.utils import _get_act, add_padding3, remove_padding3, get_grid3d
from models.basics import SpectralConv3d, LaplaceConv3d

class LaplaceBlock3d(nn.Module):
    def __init__(self, width, modes_z, modes_x, modes_y, act):
        super().__init__()

        self.laplace = LaplaceConv3d(width, width, modes_z, modes_x, modes_y)
        self.skip = nn.Conv3d(width, width, kernel_size = 1)
        self.act = _get_act(act)

    def forward(self, x):

        return self.act(self.laplace(x) + self.skip(x))


class FourierBlock3d(nn.Module):
    def __init__(self, width, modes_z, modes_x, modes_y, act):
        super().__init__()

        self.fourier = SpectralConv3d(width, width, modes_z, modes_x, modes_y)
        self.skip = nn.Conv3d(width, width, kernel_size = 1)
        self.act = _get_act(act)

    def forward(self, x):

        return self.act(self.fourier(x) + self.skip(x))



class LNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16,in_dim=4, out_dim=1, pad_ratio=0.1, act="gelu"):
        super().__init__()

        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad_ratio = pad_ratio

        self.n_layers = min(len(modes1), len(modes2), len(modes3))

        if self.n_layers == 0:
            raise ValueError("modes1, modes2 and modes3 must not be empty.")

        self.input_projection = nn.Linear(in_dim, width)

        self.laplace_layers = nn.ModuleList()
        self.fourier_layers = nn.ModuleList()

        for i in range(self.n_layers):
            self.laplace_layers.append(
                LaplaceBlock3d(
                    width=width,
                    modes_z=modes1[i],
                    modes_x=modes2[i],
                    modes_y=modes3[i],
                    act=act
                )
            )

            self.fourier_layers.append(
                FourierBlock3d(
                    width=width,
                    modes_z=modes1[i],
                    modes_x=modes2[i],
                    modes_y=modes3[i],
                    act=act
                )
            )


        self.laplace_mix = nn.Conv3d(width, width, kernel_size=1)
        self.fourier_mix = nn.Conv3d(width, width, kernel_size=1)

        self.output_projection = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, out_dim, kernel_size=1)
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"LNO3d expects [B, C, Z, X, Y], got {tuple(x.shape)}")

        B, C, Z, X, Y = x.shape

        if C != 1:
            raise ValueError(f"LNO3d for ALMA expects one input field channel, got {C}")

        grid = get_grid3d(x.shape, x.device)
        x = torch.cat([x, grid], dim=1)

        x = x.permute(0, 2, 3, 4, 1)

        x = self.input_projection(x)

        x = x.permute(0, 4, 1, 2, 3)

        if self.pad_ratio > 0:
            pad_z = int(Z * self.pad_ratio)
            pad_x = int(X * self.pad_ratio)
            pad_y = int(Y * self.pad_ratio)

            num_pad_z = (0, pad_z)
            num_pad_x = (0, pad_x)
            num_pad_y = (0, pad_y)

            x = add_padding3(x, num_pad_x, num_pad_y, num_pad_z)

        else:
            num_pad_z = (0, 0)
            num_pad_x = (0, 0)
            num_pad_y = (0, 0)

        x_laplace = x

        for layer in self.laplace_layers:
            x_laplace = layer(x_laplace)

        x_fourier = x

        for layer in self.fourier_layers:
            x_fourier = layer(x_fourier)

        x = (self.laplace_mix(x_laplace) + self.fourier_mix(x_fourier))

        x = remove_padding3(x, num_pad_x, num_pad_y, num_pad_z)

        x = self.output_projection(x)

        return x