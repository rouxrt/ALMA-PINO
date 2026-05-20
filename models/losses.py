import torch
import torch.nn as nn
import torch.fft

class PILoss(nn.Module):
    """
    Physics-Informed Loss for Optical Deconvolution.
    Combine Data Loss (MSE) with Physics Loss (Fredholm Integral Equation through FFT).
    """
    def __init__(self, lambda_data=1.0, lambda_phys=1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(self, pred_clean, dirty_image, clean_gt, psf):
        """
        pred_clean: Predicted output [Batch, Channels, X, Y]
        dirty_image: Dirty original input (from telescope) [Batch, Channels, X, Y]
        clean_gt: Clean Ground Truth [Batch, Channels, X, Y]
        psf: Telescope kernel (Dirty Beam) [Batch, Channels, X, Y]
        """
        
        loss_data = self.mse(pred_clean, clean_gt)

        if self.lambda_phys == 0.0:
            return loss_data, loss_data, torch.tensor(0.0, device=pred_clean.device)

        
        # Astronomical PSFs (e.g., from ALMA) are typically generated with the brightest 
        # peak at the geometric center of the image. However, the FFT algorithm assumes 
        # the spatial origin (0,0) is located at the top-left corner (index [0,0]).
        # 
        # If we compute the FFT on a centered PSF, the Convolution Theorem will induce 
        # an unwanted linear phase shift, causing the resulting image to translate by 
        # half the grid size (circular shift).
        # 
        # To prevent this, we use `ifftshift` to "roll" the PSF, wrapping the central 
        # peak to the top-left corner before moving into the frequency domain.
        psf_shifted = torch.fft.ifftshift(psf, dim=(-2, -1))
        
        pred_fft = torch.fft.rfft2(pred_clean)
        psf_fft = torch.fft.rfft2(psf_shifted)
        
        simulated_dirty_fft = pred_fft * psf_fft
        
        # s=pred_clean.shape[-2:] ensure that it returns to its original shape
        simulated_dirty = torch.fft.irfft2(simulated_dirty_fft, s=pred_clean.shape[-2:])
        
        loss_phys = self.mse(simulated_dirty, dirty_image)

        loss_total = (self.lambda_data * loss_data) + (self.lambda_phys * loss_phys)

        return loss_total, loss_data, loss_phys


if __name__ == "__main__":
    batch_size, canali, size = 4, 16, 32
    
    pred_clean = torch.randn(batch_size, canali, size, size, requires_grad=True)
    dirty_image = torch.randn(batch_size, canali, size, size)
    clean_gt = torch.randn(batch_size, canali, size, size)
    
    psf = torch.randn(batch_size, canali, size, size)
    
    loss_fn_data = PILoss(lambda_data=1.0, lambda_phys=0.0)
    tot1, data1, phys1 = loss_fn_data(pred_clean, dirty_image, clean_gt, psf)
    print(f"Test Only Data -> Total: {tot1.item():.4f}, Physics: {phys1.item():.4f}")
    
    loss_fn_phys = PILoss(lambda_data=1.0, lambda_phys=0.1)
    tot2, data2, phys2 = loss_fn_phys(pred_clean, dirty_image, clean_gt, psf)
    print(f"Test PI-FNO    -> Total: {tot2.item():.4f}, Physics: {phys2.item():.4f}")
    
    tot2.backward()
    print("Test done.")