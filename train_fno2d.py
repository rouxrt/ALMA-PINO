import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import optuna

from dataset.mock_dataset import MockGalaxyDatacubeDataset
from dataset.ALMA_dataset import ALMADataset
from models.fno2d import FNO2d
from models.losses import DataLoss
from models.utils import Logger, set_seed
from visualize.plot import save_predictions_FNO, plot_loss_history_FNO, visualize_datacube, plot_spectral_profile
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, tuning_mode=False):
    model.train()
    running_total_loss = 0.0
    running_l1_loss = 0.0
    running_msssim_loss = 0.0

    for batch_idx, (dirty, clean, psf) in enumerate(dataloader):
        dirty = dirty.to(device)
        clean = clean.to(device)
        psf = psf.to(device)

        optimizer.zero_grad()

        pred_clean = model(dirty)

        if (epoch % 10 == 0 and batch_idx == 0) and not tuning_mode:
            min_val = pred_clean.min().item()
            neg_percent = (pred_clean < 0).float().mean().item() * 100
            print(f"  [Debug] Min value: {min_val:.6f} | Negative pixels: {neg_percent:.1f}%")

        loss_total, l1, msssim = criterion(pred_clean, dirty, clean, psf)

        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_total_loss += loss_total.item()
        running_l1_loss += l1.item()
        running_msssim_loss += msssim.item()
    num_batches = len(dataloader)
    return (running_total_loss / num_batches,
            running_l1_loss / num_batches,
            running_msssim_loss / num_batches)

def evaluate_model(model, dataloader, criterion, device, show_datacube=False):
    model.eval() 
    running_total_loss = 0.0
    running_l1_raw = 0.0
    running_msssim = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_flux_error = 0.0
    total_samples = 0

    with torch.no_grad(): 
        for dirty, clean, psf in dataloader:
            dirty = dirty.to(device)
            clean = clean.to(device)
            psf = psf.to(device)

            raw_pred = model(dirty)

            pred_clean = torch.clamp(raw_pred, min=0.0)

            if show_datacube:
                visualize_datacube(dirty, clean, pred_clean, output_dir="results_FNO2D/visualizations", model_name="FNO2D")
                plot_spectral_profile(clean, pred_clean, sample_idx=0, output_dir="results_FNO2D/", model_name="FNO2D")
                show_datacube = False
            loss_total, loss_data, l1, msssim, loss_phys = criterion(pred_clean, dirty, clean, psf)

            running_total_loss += loss_total.item()
            running_l1_raw += l1.item()
            running_msssim += msssim.item()


            for i in range(clean.size(0)):
                sample_clean = clean[i]
                sample_pred = pred_clean[i]

                sample_max = sample_clean.max()
                if sample_max > 0:
                    p = psnr(sample_pred.unsqueeze(0), sample_clean.unsqueeze(0), data_range=sample_max)
                    s = ssim(sample_pred.unsqueeze(0), sample_clean.unsqueeze(0), data_range=sample_max)
                    running_psnr += p.item()
                    running_ssim += s.item()

                    mask = sample_clean > (0.01 * sample_max)

                    true_flux = sample_clean[mask].sum()
                    pred_flux = sample_pred[mask].sum()

                    if true_flux > 0:
                        err = torch.abs(pred_flux - true_flux) / true_flux
                        running_flux_error += err.item() * 100
                        
                    else:
                        running_flux_error += 0.0
                
                total_samples += 1
                
    num_batches = len(dataloader)
    return (running_total_loss / num_batches, 
            running_l1_raw / num_batches,
            running_msssim / num_batches,
            running_psnr / total_samples if total_samples > 0 else 0.0,
            running_ssim / total_samples if total_samples > 0 else 0.0,
            running_flux_error / total_samples if total_samples > 0 else 0.0)

def main(args):
    set_seed(42)
    os.makedirs('results_FNO2D', exist_ok=True)
    tuning_mode = hasattr(args, 'trial') and args.trial is not None

    if not tuning_mode:
        sys.stdout = Logger(f"results_FNO2D/training_log.txt")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if not tuning_mode:
        print(f"Starting training on device: {device}")
        print(f"GPU Name: {name_device}")

    history_loss = {"train": [], "val": [], "train_l1": [], "train_msssim": []}
    
    if not tuning_mode and args.mock:
        print("Loading Mock Dataset...")

    if args.mock:
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
    
    else:

        full_dataset = ALMADataset(args.dataset_path)
        
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42) 
        )

        if not tuning_mode:
            print(f"Dataset split: {train_size} Train, {val_size} Val, {test_size} Test.")

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if not tuning_mode:
        print("Initializing Fourier Neural Operator...")
    
    num_fourier_layers = args.fourier_layers
    
    model = FNO2d(
        modes1=[args.modes] * num_fourier_layers,  
        modes2=[args.modes] * num_fourier_layers,  
        width=args.width, 
        in_dim=args.channels + 2, 
        out_dim=args.channels,
        act=args.act
    ).to(device)

    criterion = DataLoss(lambda_data=args.lambda_data, alpha=args.alpha, channels=args.channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_val_flux_error = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = os.path.join('checkpoints', 'fno2d.pth')


    if not tuning_mode:
        print("Starting training loop...\n")
    for epoch in range(1, args.epochs + 1):

        tot_loss, l1, msssim = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch, tuning_mode)
        val_tot, val_l1_raw, val_msssim, val_psnr, val_ssim, val_flux = evaluate_model(model, val_dataloader, criterion, device)

        scheduler.step()
        
        if not tuning_mode:
            print(f"Epoch [{epoch}/{args.epochs}] | "
                  f"Train Loss: {tot_loss:.5f} (L1: {l1:.5f} | SSIM: {msssim:.5f}) |  "
                  f"Val Loss: {val_tot:.5f} (L1: {val_l1_raw:.5f} | SSIM: {val_msssim:.5f})")
            
        history_loss["train"].append(tot_loss)
        history_loss["val"].append(val_tot)
        history_loss["train_l1"].append(args.lambda_data * (1 - args.alpha) * l1)
        history_loss["train_msssim"].append(args.lambda_data * args.alpha * msssim)

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            best_val_l1_raw = val_l1_raw
            best_val_flux_error = val_flux
            torch.save(model.state_dict(), best_model_path)
            if not tuning_mode:
                print(f"New best model saved with Val Loss: {best_val_loss:.5f}")
        
        if (epoch % 5 == 0 or epoch == args.epochs) and not tuning_mode:
            sample_dirty, sample_clean, sample_psf = next(iter(val_dataloader))
            model.eval()
            with torch.no_grad():
                sample_pred = model(sample_dirty.to(device))
            
            save_predictions_FNO(sample_dirty, sample_clean, sample_pred, 
                                 epoch, tot_loss, l1, msssim, output_dir="results_FNO2D/predictions", dim="2D")
    
    if not tuning_mode:
        plot_loss_history_FNO(history_loss, title="FNO2D Training Loss", save_path="results_FNO2D/loss_history.png", dim="2D")

        print("\nTraining Completed!")
        print("\n" + "="*50)
        print("Evaluating best model on test set...")

        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_tot, test_l1_raw, test_msssim, test_psnr, test_ssim, test_flux_error = evaluate_model(model, test_dataloader, criterion, device, show_datacube=not tuning_mode)

        print(f"Test Loss: {test_tot:.5f} (Data: {test_l1_raw:.5f} | Phys: {test_msssim:.5f})")
        print(f"Test PSNR: {test_psnr:.5f} dB")
        print(f"Test SSIM: {test_ssim:.5f}")
        print(f"Test Flux Error: {test_flux_error:.5f}%")

    return best_val_l1_raw, best_val_flux_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training FNO2D for ALMA data")
    
    parser.add_argument('--dataset_path', type=str, default='dataset/simulations', help='Path to the dataset directory')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset instead of ALMA dataset')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples in the dataset')
    parser.add_argument('--channels', type=int, default=16, help='Number of frequency slices')
    parser.add_argument('--img_size', type=int, default=32, help='Spatial dimension XY')
    parser.add_argument('--extended_source', action='store_true', help='If active, generates only extended sources')
    
    parser.add_argument('--modes', type=int, default=8, help='Number of Fourier frequencies to keep')
    parser.add_argument('--width', type=int, default=32, help='Latent dimension of the model')
    parser.add_argument('--fourier_layers', type=int, default=4, help='Number of Fourier Layers in the model')
    
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    
    parser.add_argument('--lambda_data', type=float, default=1.0, help='Weight of the Data Loss')
    parser.add_argument('--alpha', type=float, default=0.03, help='Weighting factor for combining L1 and MS-SSIM in the data loss')
    parser.add_argument('--act', type=str, default='gelu', choices=['gelu', 'relu', 'tanh', 'leaky_relu'], help='Activation function')

    args = parser.parse_args()
    
    main(args)