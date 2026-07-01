import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset.mock_dataset import MockGalaxyDatacubeDataset
from models.fno3d import FNO3d
from models.losses import CombinedLoss
from models.utils import Logger, set_seed
from visualize.plot import save_predictions, plot_loss_history, visualize_datacube
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_total_loss = 0.0
    running_data_loss = 0.0
    running_phys_loss = 0.0
    running_l1_loss = 0.0
    running_msssim_loss = 0.0
    running_spec_loss = 0.0

    for batch_idx, (dirty, clean, psf) in enumerate(dataloader):
        dirty = dirty.to(device) 
        clean = clean.to(device)  
        psf = psf.to(device)

        optimizer.zero_grad()

        # Trasform from [B, Z, X, Y] to [B, 1, Z, X, Y] in order to add a feature channel
        dirty_3d = dirty.unsqueeze(1)
        
        pred_clean_3d = model(dirty_3d)  

        pred_clean = pred_clean_3d.squeeze(1) #remove feature channel in order to calculate the loss function

        if epoch % 10 == 0 and batch_idx == 0:
            min_val = pred_clean.min().item()
            neg_percent = (pred_clean < 0).float().mean().item() * 100
            print(f"   [Debug 3D] Min value: {min_val:.6f} | Negative pixels: {neg_percent:.1f}%")

        loss_total, loss_data, l1, msssim, loss_phys, loss_spec = criterion(pred_clean, dirty, clean, psf)

        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_total_loss += loss_total.item()
        running_data_loss += loss_data.item()
        running_phys_loss += loss_phys.item()
        running_l1_loss += l1.item()
        running_msssim_loss += msssim.item()
        running_spec_loss += loss_spec.item()
    
    num_batches = len(dataloader)
    return (running_total_loss / num_batches, 
            running_data_loss / num_batches,
            running_l1_loss / num_batches,
            running_msssim_loss / num_batches, 
            running_phys_loss / num_batches,
            running_spec_loss / num_batches)


def evaluate_model(model, dataloader, criterion, device, show_datacube=False):
    model.eval() 
    running_total_loss = 0.0
    running_data_loss = 0.0
    running_phys_loss = 0.0
    running_spec_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_flux_error = 0.0

    with torch.no_grad(): 
        for dirty, clean, psf in dataloader:
            dirty = dirty.to(device)
            clean = clean.to(device)
            psf = psf.to(device)

            dirty_3d = dirty.unsqueeze(1)
            raw_pred_3d = model(dirty_3d)
            
            pred_clean = torch.clamp(raw_pred_3d.squeeze(1), min=0.0)

            if show_datacube:
                visualize_datacube(dirty, clean, pred_clean, output_dir="results_3d/datacube_visualization")
                show_datacube = False

            loss_total, loss_data, l1, msssim, loss_phys, loss_spec = criterion(pred_clean, dirty, clean, psf)

            running_total_loss += loss_total.item()
            running_data_loss += loss_data.item()
            running_phys_loss += loss_phys.item()
            running_spec_loss += loss_spec.item()

            batch_max = clean.max().item()
            if batch_max > 0:
                batch_psnr = psnr(pred_clean, clean, data_range=batch_max)
                batch_ssim = ssim(pred_clean, clean, data_range=batch_max)
                running_psnr += batch_psnr.item()
                running_ssim += batch_ssim.item()

            flux_pred = pred_clean.sum(dim=(1,2,3))
            flux_clean = clean.sum(dim=(1,2,3))
            flux_error = torch.abs(flux_pred - flux_clean) / (flux_clean + 1e-8)
            running_flux_error += flux_error.mean().item() * 100

    num_batches = len(dataloader)
    return (running_total_loss / num_batches, 
            running_data_loss / num_batches, 
            running_phys_loss / num_batches,
            running_spec_loss / num_batches,
            running_psnr / num_batches,
            running_ssim / num_batches,
            running_flux_error / num_batches)

def main(args):
    set_seed(42)
    os.makedirs('results_3d', exist_ok=True)
    sys.stdout = Logger(f"results_3d/training_log.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting FNO3d training on device: {device}")
    print(f"GPU Name: {name_device}")

    history_loss = {
        "train": [], "val": [], "train_data": [], "train_l1": [], 
        "train_msssim": [], "train_phys": [], "train_spec": [], 
        "val_data": [], "val_phys": [], "val_spec": []
    }

    print("Loading Mock Dataset for 3D processing...")
    train_dataset = MockGalaxyDatacubeDataset(
        num_samples=args.num_samples, 
        channels=args.channels, 
        size=args.img_size,
        extended_source=args.extended_source
    )
    val_dataset = MockGalaxyDatacubeDataset(
        num_samples=args.num_samples // 5,
        channels=args.channels,
        size=args.img_size,
        extended_source=args.extended_source
    )
    test_dataset = MockGalaxyDatacubeDataset(
        num_samples=args.num_samples // 5,
        channels=args.channels,
        size=args.img_size,
        extended_source=args.extended_source
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing Spatial-Spectral Fourier Neural Operator 3D...")
    num_fourier_layers = args.fourier_layers
    
    model = FNO3d(
        modes1=[args.modes_z] * num_fourier_layers,  
        modes2=[args.modes_x] * num_fourier_layers,  
        modes3=[args.modes_y] * num_fourier_layers,  
        width=args.width, 
        in_dim=4,             
        out_dim=1,            
        pad_ratio=args.pad_ratio
    ).to(device)

    criterion = CombinedLoss(
        lambda_data=args.lambda_data, 
        lambda_phys=args.lambda_phys, 
        lambda_spec=args.lambda_spec, 
        alpha=args.alpha, 
        channels=args.channels
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = os.path.join('checkpoints', 'fno3d.pth')

    print("Starting 3D training loop...\n")
    for epoch in range(1, args.epochs + 1):

        tot_loss, data_loss, l1, msssim, phys_loss, loss_spec = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch
        )
        val_tot, val_data, val_phys, val_spec, _, _, _ = evaluate_model(
            model, val_dataloader, criterion, device
        )

        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {tot_loss:.5f} (Data: {data_loss:.5f} | Phys: {phys_loss:.5f} | Spec: {loss_spec:.5f}) |  "
              f"Val Loss: {val_tot:.5f} (Data: {val_data:.5f} | Phys: {val_phys:.5f} | Spec: {val_spec:.5f})")
        
        history_loss["train"].append(tot_loss)
        history_loss["val"].append(val_tot)
        history_loss["train_data"].append(args.lambda_data * data_loss)
        history_loss["train_l1"].append(args.lambda_data * (1 - args.alpha) * l1)
        history_loss["train_msssim"].append(args.lambda_data * args.alpha * msssim)
        history_loss["train_phys"].append(args.lambda_phys * phys_loss)
        history_loss["train_spec"].append(args.lambda_spec * loss_spec)
        history_loss["val_data"].append(args.lambda_data * val_data)
        history_loss["val_phys"].append(args.lambda_phys * val_phys)
        history_loss["val_spec"].append(args.lambda_spec * val_spec)

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            torch.save(model.state_dict(), best_model_path)
            print(f"New best 3D model saved with Val Loss: {best_val_loss:.5f}")
        
        if epoch % 5 == 0 or epoch == args.epochs:
            sample_dirty, sample_clean, sample_psf = next(iter(val_dataloader))
            model.eval()
            with torch.no_grad():
                sample_dirty_3d = sample_dirty.to(device).unsqueeze(1)
                sample_pred_3d = model(sample_dirty_3d)
                sample_pred = sample_pred_3d.squeeze(1)
            
            save_predictions(sample_dirty, sample_clean, sample_pred.cpu(), 
                             epoch, tot_loss, data_loss, phys_loss, output_dir="results_3d/predictions")
            
    plot_loss_history(history_loss, title="PI-FNO3d Training Loss", save_path="results_3d/loss_history.png")

    print("\n3D Training Completed!")
    print("\n" + "="*50)
    print("Evaluating best 3D model on test set...")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_tot, test_data, test_phys, test_spec, test_psnr, test_ssim, test_flux_error = evaluate_model(
        model, test_dataloader, criterion, device, show_datacube=True
    )

    print(f"Test Loss: {test_tot:.5f} (Data: {test_data:.5f} | Phys: {test_phys:.5f} | Spec: {test_spec:.5f})")
    print(f"Test PSNR: {test_psnr:.5f} dB")
    print(f"Test SSIM: {test_ssim:.5f}")
    print(f"Test Flux Error: {test_flux_error:.5f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training PI-FNO3d for ALMA Datacubes")
    
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--channels', type=int, default=16, help='Z dimension (frequency slices)')
    parser.add_argument('--img_size', type=int, default=32, help='Spatial dimension XY')
    parser.add_argument('--extended_source', action='store_true', help='Generates extended sources')
    
    parser.add_argument('--modes_x', type=int, default=8, help='Max Fourier modes on X axis')
    parser.add_argument('--modes_y', type=int, default=8, help='Max Fourier modes on Y axis')
    parser.add_argument('--modes_z', type=int, default=6, help='Max Fourier modes on Z/Freq axis')
    
    parser.add_argument('--width', type=int, default=16, help='Latent dimension (width)')
    parser.add_argument('--fourier_layers', type=int, default=4, help='Number of Spectral Convolution Layers')
    parser.add_argument('--pad_ratio', type=float, default=0.1, help='Padding ratio for 3D boundary protection')
    
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    
    parser.add_argument('--lambda_data', type=float, default=1.0, help='Weight of the Data Loss')
    parser.add_argument('--lambda_phys', type=float, default=0.5, help='Weight of the Physics Loss')
    parser.add_argument('--lambda_spec', type=float, default=0.0, help='Weight of the Spectral Loss')
    parser.add_argument('--alpha', type=float, default=0.03, help='Weighting factor between L1 and MS-SSIM')

    args = parser.parse_args()
    main(args)