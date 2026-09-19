import torch 
import torch.nn as nn
import torch.nn.functional as F

def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", a, b)

def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

class SpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1/(in_channels*out_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x, dim = (-2, -1))
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim = (-2, -1))
        return x
    

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        
        z_dim = min(x_ft.shape[4], self.modes3)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], self.modes3, device=x.device, dtype=torch.cfloat)
        
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding 
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, :] = compl_mul3d(coeff, self.weights1)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, :] = compl_mul3d(coeff, self.weights2)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, :] = compl_mul3d(coeff, self.weights3)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, :] = compl_mul3d(coeff, self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
        return x


class LaplaceConv2d(nn.Module):
    """
    2D Laplace pole-residue convolution.

    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H, W]

    Stable poles: Re(lambda) < 0 enforced via softplus.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2

        scale = 1.0 / (in_channels * out_channels) ** 0.5

        # Stable complex poles — Re < 0 via softplus reparameterization
        self.pole_x_real = nn.Parameter(torch.randn(out_channels, modes1) * 0.1)
        self.pole_x_imag = nn.Parameter(torch.randn(out_channels, modes1) * scale)
        self.pole_y_real = nn.Parameter(torch.randn(out_channels, modes2) * 0.1)
        self.pole_y_imag = nn.Parameter(torch.randn(out_channels, modes2) * scale)

        # Complex residues [Cin, Cout, M1, M2]
        self.weights_residue = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat)
        )

    def get_poles(self):
        pole_x = -F.softplus(self.pole_x_real) + 1j * self.pole_x_imag
        pole_y = -F.softplus(self.pole_y_real) + 1j * self.pole_y_imag
        return pole_x, pole_y

    def output_PR(self, lambda_x, lambda_y, alpha):
        """
        Pole-residue transfer function applied to truncated spectrum.

        alpha : [B, Cin, M1, M2]
        out   : [B, Cout, M1, M2]
        """
        pole_x, pole_y = self.get_poles()

        # diff_x: [Cout, M1, 1],  diff_y: [Cout, 1, M2]
        diff_x = pole_x.unsqueeze(-1) - lambda_x.view(1, -1, 1)
        diff_y = pole_y.unsqueeze(-2) - lambda_y.view(1, 1, -1)

        denominator = diff_x * diff_y + 1e-6   # numerical safety
        transfer     = 1.0 / denominator        # [Cout, M1, M2]

        # kernel: [Cin, Cout, M1, M2]
        kernel = self.weights_residue * transfer.unsqueeze(0)

        # [B, Cout, M1, M2]
        return torch.einsum("bimn,iomn->bomn", alpha, kernel)

    def forward(self, x):
        B, _, H, W = x.shape

        # 1. Truncated 2D FFT
        alpha = torch.fft.fft2(x)
        m1    = min(self.modes1, H)
        m2    = min(self.modes2, W)
        alpha_truncated = alpha[:, :, :m1, :m2]

        # 2. Laplace frequencies (imaginary axis)
        omega_x = (torch.fft.fftfreq(H, d=1.0/H, device=x.device)[:m1]
                   * 2.0 * torch.pi * 1j)
        omega_y = (torch.fft.fftfreq(W, d=1.0/W, device=x.device)[:m2]
                   * 2.0 * torch.pi * 1j)

        # 3. Pole-residue response: [B, Cout, M1, M2]
        out_ft = self.output_PR(omega_x, omega_y, alpha_truncated)

        # 4. Laplace basis reconstruction
        pole_x, pole_y = self.get_poles()

        grid_x = torch.linspace(0.0, 1.0, H, device=x.device,
                                dtype=torch.float32)
        grid_y = torch.linspace(0.0, 1.0, W, device=x.device,
                                dtype=torch.float32)

        # basis_x: [Cout, M1, H],  basis_y: [Cout, M2, W]
        basis_x = torch.exp(torch.einsum("om,s->oms", pole_x, grid_x))
        basis_y = torch.exp(torch.einsum("on,t->ont", pole_y, grid_y))

        # [B, Cout, H, W]
        out = torch.einsum("bomn,omh,onw->bohw", out_ft, basis_x, basis_y)

        return (out / (H * W)).real


class LaplaceConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_z, modes_x, modes_y):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_x = modes_x
        self.modes_y = modes_y

        self.pole_z_real = nn.Parameter(torch.randn(out_channels, modes_z) * 0.1)
        self.pole_z_imag = nn.Parameter(torch.randn(out_channels, modes_z) * 0.1)
        self.pole_x_real = nn.Parameter(torch.randn(out_channels, modes_x) * 0.1)
        self.pole_x_imag = nn.Parameter(torch.randn(out_channels, modes_x) * 0.1)
        self.pole_y_real = nn.Parameter(torch.randn(out_channels, modes_y) * 0.1)
        self.pole_y_imag = nn.Parameter(torch.randn(out_channels, modes_y) * 0.1)

        scale = 1.0 / (in_channels * modes_z * modes_x * modes_y) ** 0.5

        self.residues = nn.Parameter(torch.randn(in_channels, out_channels, modes_z, modes_x, modes_y, dtype=torch.cfloat) * scale)

    def forward(self, x):
        B, _, Z, X, Y = x.shape

        mz = min(self.modes_z, Z)
        mx = min(self.modes_x, X)
        my = min(self.modes_y, Y)

        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))

        x_ft = x_ft[:, :, :mz, :mx, :my]

        pole_z = -F.softplus(self.pole_z_real[:, :mz]) + 1j * self.pole_z_imag[:, :mz]
        pole_x = -F.softplus(self.pole_x_real[:, :mx]) + 1j * self.pole_x_imag[:, :mx]
        pole_y = -F.softplus(self.pole_y_real[:, :my]) + 1j * self.pole_y_imag[:, :my]

        omega_z = torch.fft.fftfreq(Z, d=1.0 / Z, device=x.device)[:mz]
        omega_x = torch.fft.fftfreq(X, d=1.0 / X, device=x.device)[:mx]
        omega_y = torch.fft.fftfreq(Y, d=1.0 / Y, device=x.device)[:my]

        omega_z = 2j * torch.pi * omega_z
        omega_x = 2j * torch.pi * omega_x   
        omega_y = 2j * torch.pi * omega_y


        diff_z = (pole_z[:, :, None, None] - omega_z[None, :, None, None])
        diff_x = (pole_x[:, None, :, None] - omega_x[None, None, :, None])
        diff_y = (pole_y[:, None, None, :] - omega_y[None, None, None, :])

        denominator = (diff_z * diff_x * diff_y)
        transfer_function = 1.0 / (denominator + 1e-6)

        residues = self.residues[:, :, :mz, :mx, :my]

        kernel = (residues * transfer_function.unsqueeze(0))

        out_ft = torch.einsum("bizxy,iozxy->bozxy", x_ft, kernel)

        grid_z = torch.linspace(0, 1, Z, device=x.device)
        grid_x = torch.linspace(0, 1, X, device=x.device)
        grid_y = torch.linspace(0, 1, Y, device=x.device)

        basis_z = torch.exp(pole_z.unsqueeze(-1) * grid_z.view(1, 1, Z))
        basis_x = torch.exp(pole_x.unsqueeze(-1) * grid_x.view(1, 1, X))
        basis_y = torch.exp(pole_y.unsqueeze(-1) * grid_y.view(1, 1, Y))

        out = torch.einsum("bozxy,ozZ,oxX,oyY->boZXY", out_ft, basis_z, basis_x, basis_y)

        out = out / float(Z * X * Y)

        return out.real