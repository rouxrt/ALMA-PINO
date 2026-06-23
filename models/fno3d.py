import torch
import torch.nn as nn
from .basics import SpectralConv3d
from .utils import _get_act, add_padding3, remove_padding3, get_grid3d


class FNO3d(nn.Module):
    def __init__(self, 
                 modes1, modes2, modes3,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=0.0):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch_size, channels, size_z, size_x, size_y) 
        Returns:
            - x: (batch_size, out_dim, size_z, size_x, size_y)
        '''

        batchsize = x.shape[0]

        grid = get_grid3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        size_z, size_x, size_y = x.shape[2], x.shape[3], x.shape[4]

        if self.pad_ratio > 0:
            num_pad1 = [0, round(self.pad_ratio * size_x)] 
            num_pad2 = [0, round(self.pad_ratio * size_y)] 
            num_pad3 = [0, round(self.pad_ratio * size_z)] 
        else:
            num_pad1 = num_pad2 = num_pad3 = [0, 0]

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)

        x = x.permute(0, 4, 1, 2, 3)

        x = add_padding3(x, num_pad1=num_pad1, num_pad2=num_pad2, num_pad3=num_pad3)
        size_z_pad, size_x_pad, size_y_pad = x.shape[2], x.shape[3], x.shape[4]

        length = len(self.ws)
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_z_pad, size_x_pad, size_y_pad)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)

        x = remove_padding3(x, num_pad1=num_pad1, num_pad2=num_pad2, num_pad3=num_pad3)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        x = x.permute(0, 4, 1, 2, 3)
        return x