import torch
import torch.nn as nn
import torch.fft
from pytorch_msssim import MS_SSIM, SSIM

    
class DataLoss(nn.Module): 
    def __init__(self, lambda_data=1.0, alpha=0.84, channels=64):
        super().__init__()
        self.lambda_data = lambda_data
        self.alpha = alpha
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=channels, win_size=11)

        self.mse_phys = nn.MSELoss()

    def forward(self, pred_clean, dirty_image, clean_gt, psf):
        
        batch_max = clean_gt.max().clamp(min=1e-8)

        #normalize to [0,1] in order to have same scale for L1, SSIM and physics loss
        pred_norm = pred_clean / batch_max
        gt_norm = clean_gt / batch_max
        dirty_norm = dirty_image / batch_max

        l1_norm = self.l1_loss(pred_norm, gt_norm)

        ssim_val = self.ssim_loss(pred_norm, gt_norm)
        ssim_norm = 1.0 - ssim_val

        loss_data_norm = (self.alpha * ssim_norm) + ((1 - self.alpha) * l1_norm)

        return  loss_data_norm, l1_norm, ssim_norm

class CombinedLoss(nn.Module): 
    def __init__(self, lambda_data=1.0, lambda_phys=1.0, alpha=0.84, channels=64):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.alpha = alpha
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=channels, win_size=11)

        self.mse_phys = nn.MSELoss()

    def forward(self, pred_clean, dirty_image, clean_gt, psf):
        
        batch_max = clean_gt.max().clamp(min=1e-8)

        #normalize to [0,1] in order to have same scale for L1, SSIM and physics loss
        pred_norm = pred_clean / batch_max
        gt_norm = clean_gt / batch_max
        dirty_norm = dirty_image / batch_max

        l1_norm = self.l1_loss(pred_norm, gt_norm)

        ssim_val = self.ssim_loss(pred_norm, gt_norm)
        ssim_norm = 1.0 - ssim_val

        loss_data_norm = (self.alpha * ssim_norm) + ((1 - self.alpha) * l1_norm)

        psf_shifted = torch.fft.ifftshift(psf, dim=(-2, -1))
        pred_fft = torch.fft.rfft2(pred_norm)
        psf_fft = torch.fft.rfft2(psf_shifted)
        simulated_dirty_fft = pred_fft * psf_fft
        simulated_dirty_norm = torch.fft.irfft2(simulated_dirty_fft, s=pred_norm.shape[-2:])
        
        loss_phys_norm = self.mse_phys(simulated_dirty_norm, dirty_norm) 

        loss_total_norm = (self.lambda_data * loss_data_norm) + (self.lambda_phys * loss_phys_norm)

        return loss_total_norm, loss_data_norm, l1_norm, ssim_norm, loss_phys_norm


# class CombinedLoss(nn.Module): 
#     def __init__(self, lambda_data=1.0, lambda_phys=1.0, lambda_spec=0.1, alpha=0.84, channels=64):
#         super().__init__()
#         self.lambda_data = lambda_data
#         self.lambda_phys = lambda_phys
#         self.lambda_spec = lambda_spec
#         self.alpha = alpha

#         self.l1_loss = nn.L1Loss()
#         self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=channels, win_size=11)

#         self.mse_phys = nn.MSELoss()
#     def forward(self, pred_clean, dirty_image, clean_gt, psf):

#         # ── Normalizzazione per-campione ──
#         sample_max = clean_gt.flatten(1).max(dim=1).values.clamp(min=1e-8)
#         sample_max = sample_max[:, None, None, None]

#         pred_norm  = pred_clean  / sample_max
#         gt_norm    = clean_gt    / sample_max
#         dirty_norm = dirty_image / sample_max

#         # ── Data loss ──
#         l1_norm   = self.l1_loss(pred_norm, gt_norm)
#         ssim_val  = self.ssim_loss(pred_norm.clamp(0, 1), gt_norm)
#         ssim_norm = 1.0 - ssim_val
#         loss_data = (self.alpha * ssim_norm) + ((1 - self.alpha) * l1_norm)

#         # ── Physics loss (invariante alla normalizzazione lineare) ──
#         psf_shifted          = torch.fft.ifftshift(psf, dim=(-2, -1))
#         pred_fft             = torch.fft.rfft2(pred_norm)
#         psf_fft              = torch.fft.rfft2(psf_shifted)
#         simulated_dirty_norm = torch.fft.irfft2(pred_fft * psf_fft, s=pred_norm.shape[-2:])
#         loss_phys            = self.mse_phys(simulated_dirty_norm, dirty_norm)

#         # ── Spectral loss (spettro integrato, non derivate) ──
#         loss_spec = self.mse_phys(
#             pred_norm.mean(dim=(-2, -1)),
#             gt_norm.mean(dim=(-2, -1))
#         )

#         loss_total = (self.lambda_data * loss_data +
#                     self.lambda_phys * loss_phys +
#                     self.lambda_spec * loss_spec)

#         return loss_total, loss_data, l1_norm, ssim_norm, loss_phys, loss_spec

