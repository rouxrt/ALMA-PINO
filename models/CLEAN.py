import torch
import torchvision.transforms.functional as TF


def hogbom_clean_batch(dirty, psf, gain=0.1, n_iter=200, threshold=1e-3):
    """ Esegue CLEAN iterativo su un batch di datacube nel dominio dell'immagine. """
    B, Z, X, Y = dirty.shape
    dirty_flat = dirty.view(B*Z, X, Y)
    psf_flat = psf.view(B*Z, X, Y)
    
    clean_model = torch.zeros_like(dirty_flat)
    residual = dirty_flat.clone()
    cx, cy = X // 2, Y // 2
    
    psf_max = psf_flat.amax(dim=(1,2), keepdim=True)
    psf_max[psf_max == 0] = 1.0
    psf_flat = psf_flat / psf_max

    for _ in range(n_iter):
        max_vals, max_idxs = residual.view(B*Z, -1).max(dim=1)
        active_mask = max_vals > threshold
        
        if not active_mask.any():
            break
            
        mx = max_idxs // Y
        my = max_idxs % Y
        
        for i in range(B*Z):
            if active_mask[i]:
                val = gain * max_vals[i]
                clean_model[i, mx[i], my[i]] += val
                
                shift_x = mx[i] - cx
                shift_y = my[i] - cy
                shifted_psf = torch.roll(psf_flat[i], shifts=(shift_x.item(), shift_y.item()), dims=(0, 1))
                residual[i] -= val * shifted_psf

    clean_model = clean_model.view(B, Z, X, Y)
    restored = TF.gaussian_blur(clean_model, kernel_size=[3, 3], sigma=[1.0, 1.0]) 
    return restored