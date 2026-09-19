import torch
import torch.nn as nn

from .utils import _get_act, add_padding2, remove_padding2, get_grid2d
from .basics import SpectralConv2D, LaplaceConv2d



class LaplaceBlock2d(nn.Module):
    """
    Single LNO layer:  LaplaceConv2d(x) + Conv2d_1x1(x)  →  act
    """

    def __init__(self, width, modes1, modes2, act):
        super().__init__()

        self.laplace = LaplaceConv2d(width, width, modes1, modes2)
        self.skip    = nn.Conv2d(width, width, kernel_size=1)
        self.act     = _get_act(act)

    def forward(self, x):
        return self.act(self.laplace(x) + self.skip(x))


class FourierBlock2d(nn.Module):
    """
    Single FNO layer:  SpectralConv2D(x) + Conv2d_1x1(x)  →  act
    """

    def __init__(self, width, modes1, modes2, act):
        super().__init__()

        self.fourier = SpectralConv2D(width, width, modes1, modes2)
        self.skip    = nn.Conv2d(width, width, kernel_size=1)
        self.act     = _get_act(act)

    def forward(self, x):
        return self.act(self.fourier(x) + self.skip(x))



class LNO2d(nn.Module):
    def __init__(self, modes1, modes2, width = 32, in_dim = 18, out_dim = 16, pad_ratio = 0.1, act = 'gelu'):
        super().__init__()

        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad_ratio = pad_ratio

        self.n_layers = min(len(modes1), len(modes2))

        if self.n_layers == 0:
            raise ValueError("modes1 and modes2 must not be empty.")

        self.input_projection = nn.Linear(in_dim, width)

        self.laplace_layers = nn.ModuleList([
            LaplaceBlock2d(width = width, modes1 = modes1[i], modes2 = modes2[i], act = act)
            for i in range(self.n_layers)
        ])

        self.fourier_layers = nn.ModuleList([
            FourierBlock2d(width = width, modes1 = modes1[i], modes2 = modes2[i], act = act)
            for i in range(self.n_layers)
        ])

        self.laplace_mix = nn.Conv2d(width, width, kernel_size=1)
        self.fourier_mix = nn.Conv2d(width, width, kernel_size=1)

        self.output_projection = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_dim, kernel_size=1)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"LNO2d expects [B, C, H, W], got {tuple(x.shape)}")

        B, C, H, W = x.shape

        grid = get_grid2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)  

        x = x.permute(0, 2, 3, 1) 
        x = self.input_projection(x)
        x = x.permute(0, 3, 1, 2)

        if self.pad_ratio > 0:
            pad_h = (0, int(H * self.pad_ratio))
            pad_w = (0, int(W * self.pad_ratio))
            x = add_padding2(x, list(pad_h), list(pad_w))
        else:
            pad_h = (0, 0)
            pad_w = (0, 0)


        x_laplace = x
        x_fourier = x

        for laplace_layer, fourier_layer in zip(self.laplace_layers, self.fourier_layers):
            x_laplace = laplace_layer(x_laplace)
            x_fourier = fourier_layer(x_fourier)


        x = self.laplace_mix(x_laplace) + self.fourier_mix(x_fourier)

        x = remove_padding2(x, list(pad_h), list(pad_w))

        x = self.output_projection(x)

        return x