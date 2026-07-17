import os
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class ALMADataset(Dataset):
    def __init__(self, dataset_dir):
        """
        Initialize the dataset by reading the folder structure created by almasim.
        
        Args:
            dataset_dir (str): The path to the main folder (e.g., 'alma_dataset/simulations')
        """
        self.dataset_dir = Path(dataset_dir)
        self.samples = []

        for sim_folder in sorted(self.dataset_dir.glob("sim_*")):
            h5_path = sim_folder / "dataset.h5"
            if h5_path.exists():
                self.samples.append(h5_path)
        print(f"Found {len(self.samples)} valid datacubes in folder {dataset_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_file = self.samples[idx]
        
        with h5py.File(h5_file, 'r') as f:
            dirty = np.array(f['dirty_cube'])
            clean = np.array(f['clean_cube'])
            uv_mask = np.array(f['uv_mask_cube'])

        dirty_tensor = torch.tensor(dirty, dtype=torch.float32)
        clean_tensor = torch.tensor(clean, dtype=torch.float32)
        uv_mask_tensor = torch.tensor(uv_mask, dtype=torch.float32)
        
        uv_shifted = torch.fft.ifftshift(uv_mask_tensor, dim=(-2, -1))
        
        psf_complex = torch.fft.ifft2(uv_shifted, dim=(-2, -1))
        
        psf_spatial = torch.fft.fftshift(psf_complex.real, dim=(-2, -1))
        
        psf_max = psf_spatial.amax(dim=(-2, -1), keepdim=True)
        psf_tensor = psf_spatial / (psf_max + 1e-8)
        
        dirty_tensor = torch.nan_to_num(dirty_tensor, nan=0.0)
        clean_tensor = torch.nan_to_num(clean_tensor, nan=0.0)
        psf_tensor = torch.nan_to_num(psf_tensor, nan=0.0)

        return dirty_tensor, clean_tensor, psf_tensor

#TEST
if __name__ == "__main__":
    dataset_path = "dataset/simulations"
    
    if os.path.exists(dataset_path):
        my_dataset = ALMADataset(dataset_path)
        dirty, clean, psf = my_dataset[0]
       
        print("\nShape of the loaded tensors:")
        print(f"Dirty shape: {dirty.shape}")
        print(f"Clean shape: {clean.shape}")
        print(f"PSF shape:   {psf.shape}")

        sample_idx = 0
        dirty_img = dirty.detach().cpu().numpy()
        clean_img = clean.detach().cpu().numpy()
        psf_img = psf.detach().cpu().numpy()
        
        channels_to_plot = [0, 8, 15] 
        
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 8))
        fig.suptitle("Ground Truth vs Observation", fontsize=16)

        for i, ch in enumerate(channels_to_plot):
            ax_clean = axes[0, i]
            im_clean = ax_clean.imshow(clean_img[ch], cmap='magma', origin='lower')
            ax_clean.set_title(f"Clean - Canale {ch}")
            fig.colorbar(im_clean, ax=ax_clean, fraction=0.046, pad=0.04)
            
            ax_dirty = axes[1, i]
            im_dirty = ax_dirty.imshow(dirty_img[ch], cmap='magma', origin='lower')
            ax_dirty.set_title(f"Dirty - Canale {ch}")
            fig.colorbar(im_dirty, ax=ax_dirty, fraction=0.046, pad=0.04)

            ax_psf = axes[2, i]
            im_psf = ax_psf.imshow(psf_img[ch], cmap='magma', origin='lower', vmin=-0.05, vmax=0.05)
            ax_psf.set_title(f"PSF - Canale {ch}")
            fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    else:
        print("Simulations folder not found. Modify the path to test.")