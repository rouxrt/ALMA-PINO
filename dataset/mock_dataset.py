import torch
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

class MockGalaxyDatacubeDataset(Dataset):
    def __init__(self, num_samples=100, channels=16, size=32, base_sigma_psf=2.5, extended_source = False):
        super().__init__()
        self.num_samples = num_samples
        self.channels = channels
        self.size = size
        self.extended_source = extended_source
        
        #  PSF narrows at higher frequencies
        psfs = []
        for c in range(channels):
            # reducing radius of PSF as we go up in channels
            current_sigma = base_sigma_psf * (1.0 - 0.02 * c)
            psfs.append(self._create_gaussian_2d(size, current_sigma))
            
        self.psf = torch.stack(psfs, dim=0) # Shape: [16, 32, 32]
        
        # moving center of PSF to origin (0,0) before FFT to avoid circular shift issues   
        psf_shifted = torch.fft.ifftshift(self.psf, dim=(-2, -1))
        
        self.psf_fft = torch.fft.rfft2(psf_shifted) # Ora è perfetta!

    def _create_gaussian_2d(self, size, sigma_x, sigma_y=None, center_x=None, center_y=None):
        if center_x is None: center_x = size // 2
        if center_y is None: center_y = size // 2
        if sigma_y is None: sigma_y = sigma_x

        coords_x = torch.arange(size, dtype=torch.float32) - center_x
        coords_y = torch.arange(size, dtype=torch.float32) - center_y
        y, x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        
        g = torch.exp(-( (x**2)/(2 * sigma_x**2) + (y**2)/(2 * sigma_y**2) ))
        return g / g.sum()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean_image = torch.zeros(self.channels, self.size, self.size)
        
        if not self.extended_source:
            center_x = torch.randint(self.size//4, 3*self.size//4, (1,)).item()
            center_y = torch.randint(self.size//4, 3*self.size//4, (1,)).item()

            peak_c = torch.randint(0, self.channels, (1,)).item()

            for c in range(self.channels):
                spectral_intensity = math.exp(-((c - peak_c)**2) / 4.0)
                clean_image[c, center_x, center_y] = spectral_intensity

        else:
            num_sources = torch.randint(2, 5, (1,)).item()

            blob_params = []
            for _ in range(num_sources):
                cx = torch.randint(self.size//8, 7*self.size//8, (1,)).item()
                cy = torch.randint(self.size//8, 7*self.size//8, (1,)).item()
                sx = torch.rand(1).item() * 2.0 + 0.8
                sy = torch.rand(1).item() * 2.0 + 0.8
                intensity_mod = torch.rand(1).item() * 0.6 + 0.4 

                peak_c = torch.randint(0, self.channels, (1,)).item()
                
                blob_params.append((cx, cy, sx, sy, intensity_mod, peak_c))

            for c in range(self.channels):
                channel_map = torch.zeros(self.size, self.size)
                for (cx, cy, sx, sy, int_mod, peak_c) in blob_params:
                    spectral_intensity = math.exp(-((c - peak_c)**2) / 4.0)
                    blob = self._create_gaussian_2d(self.size, sigma_x=sx, sigma_y=sy, center_x=cx, center_y=cy)
                    channel_map += blob * int_mod * spectral_intensity
                
                clean_image[c] = channel_map


        clean_fft = torch.fft.rfft2(clean_image)
        dirty_fft = clean_fft * self.psf_fft
        dirty_image = torch.fft.irfft2(dirty_fft, s=(self.size, self.size))
        
        noise = torch.randn_like(dirty_image) * 0.005
        dirty_image = dirty_image + noise

        return dirty_image, clean_image, self.psf

# --- TEST ---
if __name__ == "__main__":
    dataset = MockGalaxyDatacubeDataset(channels=16, size=32, extended_source=True)
    loader = DataLoader(dataset, batch_size=4)
    
    dirty, clean, psf = next(iter(loader))
    print(f"Shape Immagine Sporca (Input Rete): {dirty.shape}")
    print(f"Shape Immagine Pulita (Loss Data): {clean.shape}")
    print(f"Shape PSF (Loss Fisica): {psf.shape}")
    print("\nDataset Astrofisico Sintetico Multi-Canale PRONTO!")

    print("Generazione dei grafici in corso...")
    
    sample_idx = 0
    dirty_img = dirty[sample_idx].detach().cpu().numpy()
    clean_img = clean[sample_idx].detach().cpu().numpy()
    
    channels_to_plot = [0, 8, 15] 
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    fig.suptitle("Simulazione Telescopio: Ground Truth vs Osservazione", fontsize=16)

    for i, ch in enumerate(channels_to_plot):
        ax_clean = axes[0, i]
        im_clean = ax_clean.imshow(clean_img[ch], cmap='magma', origin='lower')
        ax_clean.set_title(f"Clean - Canale {ch}")
        fig.colorbar(im_clean, ax=ax_clean, fraction=0.046, pad=0.04)
        
        ax_dirty = axes[1, i]
        im_dirty = ax_dirty.imshow(dirty_img[ch], cmap='magma', origin='lower')
        ax_dirty.set_title(f"Dirty - Canale {ch}")
        fig.colorbar(im_dirty, ax=ax_dirty, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()