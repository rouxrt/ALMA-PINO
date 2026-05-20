import torch
import torch.nn.functional as F

def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func


def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), 'constant', 0.)
    else:
        res = x
    return res



def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0]:-num_pad1[1], num_pad2[0]:-num_pad2[1]]
    else:
        res = x
    return res

def get_grid2d(self, shape, device):
    batchsize, size_x, size_y = shape[0], shape[2], shape[3]
    gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
    gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
    return torch.cat((gridx, gridy), dim=1).to(device)