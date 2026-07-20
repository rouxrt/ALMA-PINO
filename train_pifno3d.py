import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import optuna
import copy

from dataset.mock_dataset import MockGalaxyDatacubeDataset
from dataset.ALMA_dataset import ALMADataset
from models.fno3d import FNO3d
from models.losses import CombinedLoss
from models.utils import Logger, set_seed
from visualize.plot import save_predictions, plot_loss_history, visualize_datacube, plot_spectral_profile
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, tuning_mode=False):

    model.train()
    running_total_loss = 0.0
    running_data_loss = 0.0
    running_phys_loss = 0.0
    running_l1_loss = 0.0
    running_msssim_loss = 0.0

    for batch_idx, (dirty, clean, psf) in enumerate(dataloader):
        dirty = dirty.to(device) 
        clean = clean.to(device)  
        psf = psf.to(device)

        optimizer.zero_grad()

        # Trasform from [B, Z, X, Y] to [B, 1, Z, X, Y] in order to add a feature channel
        dirty_3d = dirty.unsqueeze(1)
        
        pred_clean_3d = model(dirty_3d)  

        pred_clean = pred_clean_3d.squeeze(1) #remove feature channel in order to calculate the loss function

        if (epoch % 10 == 0 and batch_idx == 0) and not tuning_mode:
            min_val = pred_clean.min().item()
            neg_percent = (pred_clean < 0).float().mean().item() * 100
            print(f"   [Debug 3D] Min value: {min_val:.6f} | Negative pixels: {neg_percent:.1f}%")

        loss_total, loss_data, l1, msssim, loss_phys = criterion(pred_clean, dirty, clean, psf)

        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_total_loss += loss_total.item()
        running_data_loss += loss_data.item()
        running_phys_loss += loss_phys.item()
        running_l1_loss += l1.item()
        running_msssim_loss += msssim.item()
    
    num_batches = len(dataloader)
    return (running_total_loss / num_batches, 
            running_data_loss / num_batches,
            running_l1_loss / num_batches,
            running_msssim_loss / num_batches, 
            running_phys_loss / num_batches)

def evaluate_model(model, dataloader, criterion, device, show_datacube=False):
    model.eval() 
    running_total_loss = 0.0
    running_data_loss = 0.0
    running_l1_raw = 0.0
    running_phys_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_flux_error = 0.0
    total_samples = 0

    with torch.no_grad(): 
        for dirty, clean, psf in dataloader:
            dirty = dirty.to(device)
            clean = clean.to(device)
            psf = psf.to(device)

            dirty_3d = dirty.unsqueeze(1) 
            raw_pred = model(dirty_3d).squeeze(1)

            pred_clean = torch.clamp(raw_pred, min=0.0)

            if show_datacube:
                visualize_datacube(dirty, clean, pred_clean, output_dir="results_PIFNO3D/datacube_visualization")
                plot_spectral_profile(clean, pred_clean, sample_idx=0, output_dir="results_PIFNO3D/")
                show_datacube = False
            loss_total, loss_data, l1, msssim, loss_phys = criterion(pred_clean, dirty, clean, psf)

            running_total_loss += loss_total.item()
            running_data_loss += loss_data.item()
            running_l1_raw += l1.item()
            running_phys_loss += loss_phys.item()


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
            running_data_loss / num_batches, 
            running_l1_raw / num_batches,
            running_phys_loss / num_batches,
            running_psnr / total_samples if total_samples > 0 else 0.0,
            running_ssim / total_samples if total_samples > 0 else 0.0,
            running_flux_error / total_samples if total_samples > 0 else 0.0)

def test_time_optimize(model, dirty, psf, device, tto_epochs, tto_lr, channels):

    original_state = copy.deepcopy(model.state_dict())

    tto_criterion = CombinedLoss(
        lambda_data=0.0,
        lambda_phys=1.0,
        lambda_spec=0.0,
        alpha=0.0,
        channels=channels
    ).to(device)

    tto_optimizer = optim.Adam(model.parameters(), lr=tto_lr)

    dirty_dev = dirty.to(device)
    psf_dev   = psf.to(device)
    clean_placeholder = torch.zeros_like(dirty_dev)

    model.train()
    for step in range(tto_epochs):
        tto_optimizer.zero_grad()
        pred = torch.clamp(model(dirty_dev.unsqueeze(1)).squeeze(1), min=0.0)

        loss_total, _, _, _, phys_loss = tto_criterion(
            pred, dirty_dev, clean_placeholder, psf_dev
        )
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        tto_optimizer.step()

    model.eval()
    with torch.no_grad():
        final_pred = torch.clamp(model(dirty_dev.unsqueeze(1)).squeeze(1), min=0.0)

    # Restore the original model state to avoid affecting subsequent evaluations
    model.load_state_dict(original_state)

    return final_pred


def evaluate_with_tto(model, dataloader, criterion, device, channels,
                      tto_epochs=10, tto_lr=1e-5, verbose=False):

    model.eval()

    metrics_before = {"flux": 0.0, "psnr": 0.0, "ssim": 0.0}
    metrics_after  = {"flux": 0.0, "psnr": 0.0, "ssim": 0.0}

    total_samples = 0

    with torch.no_grad():
        for dirty, clean, psf in dataloader:
            dirty = dirty.to(device)
            clean = clean.to(device)
            pred = torch.clamp(model(dirty.unsqueeze(1)).squeeze(1), min=0.0)

            for i in range(clean.size(0)):
                sc = clean[i]; sp = pred[i]
                smax = sc.max()
                if smax > 0:
                    mask = sc > (0.01 * smax)
                    err  = torch.abs(sp[mask].sum() - sc[mask].sum()) / (sc[mask].sum() + 1e-8)
                    metrics_before["flux"] += err.item() * 100
                    metrics_before["psnr"] += psnr(sp.unsqueeze(0), sc.unsqueeze(0), data_range=smax.item())
                    metrics_before["ssim"] += ssim(sp.unsqueeze(0), sc.unsqueeze(0), data_range=smax.item())
                total_samples += 1

    total_samples_tto = 0
    for batch_idx, (dirty, clean, psf) in enumerate(dataloader):
        for i in range(dirty.size(0)):
            d = dirty[i].unsqueeze(0)   # [1, C, H, W]
            c = clean[i].unsqueeze(0)
            p = psf[i].unsqueeze(0)

            pred_tto = test_time_optimize(
                model, d, p, device, tto_epochs, tto_lr, channels
            )

            sc = c[0].to(device); sp = pred_tto[0]
            smax = sc.max()
            if smax > 0:
                mask = sc > (0.01 * smax)
                err  = torch.abs(sp[mask].sum() - sc[mask].sum()) / (sc[mask].sum() + 1e-8)
                metrics_after["flux"] += err.item() * 100
                metrics_after["psnr"] += psnr(sp.unsqueeze(0), sc.unsqueeze(0), data_range=smax.item())
                metrics_after["ssim"] += ssim(sp.unsqueeze(0), sc.unsqueeze(0), data_range=smax.item())

            total_samples_tto += 1

    n = total_samples if total_samples > 0 else 1
    n_tto = total_samples_tto if total_samples_tto > 0 else 1

    return {
        "before": {k: v / n     for k, v in metrics_before.items()},
        "after":  {k: v / n_tto for k, v in metrics_after.items()},
    }

def main(args):
    set_seed(42)
    os.makedirs('results_PIFNO3D', exist_ok=True)
    tuning_mode = hasattr(args, 'trial') and args.trial is not None

    if not tuning_mode:
        sys.stdout = Logger(f"results_PIFNO3D/training_log.txt")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if not tuning_mode:
        print(f"Starting training on device: {device}")
        print(f"GPU Name: {name_device}")

    history_loss = {"train": [], "val": [], "train_data": [], "train_l1": [], "train_msssim": [], "train_phys": [], "val_data": [], "val_phys": []}
    
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
    
    model = FNO3d(
        modes1=[args.modes_z] * num_fourier_layers,  
        modes2=[args.modes_x] * num_fourier_layers,  
        modes3=[args.modes_y] * num_fourier_layers,  
        width=args.width, 
        in_dim=4,             
        out_dim=1,            
        pad_ratio=args.pad_ratio,
        act=args.act
    ).to(device)

    criterion = CombinedLoss(lambda_data=args.lambda_data, lambda_phys=args.lambda_phys, alpha=args.alpha, channels=args.channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_val_flux_error = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = os.path.join('checkpoints', 'pifno3d.pth')


    if not tuning_mode:
        print("Starting training loop...\n")
    for epoch in range(1, args.epochs + 1):

        tot_loss, data_loss, l1, msssim, phys_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch, tuning_mode)
        val_tot, val_data, val_l1_raw, val_phys, _, _, val_flux = evaluate_model(model, val_dataloader, criterion, device)

        scheduler.step()
        
        if not tuning_mode:
            print(f"Epoch [{epoch}/{args.epochs}] | "
                  f"Train Loss: {tot_loss:.5f} (Data: {data_loss:.5f} | Phys: {phys_loss:.5f}) |  "
                  f"Val Loss: {val_tot:.5f} (Data: {val_data:.5f} | Phys: {val_phys:.5f})")
            
        history_loss["train"].append(tot_loss)
        history_loss["val"].append(val_tot)
        history_loss["train_data"].append(args.lambda_data * data_loss)
        history_loss["train_l1"].append(args.lambda_data * (1 - args.alpha) * l1)
        history_loss["train_msssim"].append(args.lambda_data * args.alpha * msssim)
        history_loss["train_phys"].append(args.lambda_phys * phys_loss)
        history_loss["val_data"].append(args.lambda_data * val_data)
        history_loss["val_phys"].append(args.lambda_phys * val_phys)

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
                sample_pred = model(sample_dirty.to(device).unsqueeze(1)).squeeze(1)
            
            save_predictions(sample_dirty, sample_clean, sample_pred, 
                             epoch, tot_loss, data_loss, phys_loss, output_dir="results_PIFNO3D/predictions")
    
    if not tuning_mode:
        plot_loss_history(history_loss, title="PI-FNO Training Loss", save_path="results_PIFNO3D/loss_history.png")

        print("\nTraining Completed!")
        print("\n" + "="*50)
        print("Evaluating best model on test set...")

    if not tuning_mode:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        # Evaluation standard
        test_tot, test_data, _, test_phys, test_psnr, test_ssim, test_flux = \
            evaluate_model(model, test_dataloader, criterion, device, show_datacube=True)

        print(f"\nTest Loss : {test_tot:.5f}  (Data: {test_data:.5f} | Phys: {test_phys:.5f})")
        print(f"Test PSNR : {test_psnr:.5f} dB")
        print(f"Test SSIM : {test_ssim:.5f}")
        print(f"Test Flux Error: {test_flux:.5f}%")

        #TEST TIME OPTIMIZATION
        print("\n" + "="*50)
        print(f"Test Time Optimization | epoche={args.tto_epochs} | lr={args.tto_lr}")
        print("="*50)

        tto_results = evaluate_with_tto(
            model, test_dataloader, criterion, device,
            channels=args.channels,
            tto_epochs=args.tto_epochs,
            tto_lr=args.tto_lr,
            verbose=True,
        )

        b = tto_results["before"]
        a = tto_results["after"]

        print(f"\n{'Metrica':<18} {'Pre-TTO':>12} {'Post-TTO':>12} {'Δ':>10}")
        print("-" * 55)
        print(f"{'Flux Error (%)':<18} {b['flux']:>12.4f} {a['flux']:>12.4f} "
              f"{a['flux']-b['flux']:>+10.4f}")
        print(f"{'PSNR (dB)':<18} {b['psnr']:>12.4f} {a['psnr']:>12.4f} "
              f"{a['psnr']-b['psnr']:>+10.4f}")
        print(f"{'SSIM':<18} {b['ssim']:>12.4f} {a['ssim']:>12.4f} "
              f"{a['ssim']-b['ssim']:>+10.4f}")

    return best_val_l1_raw, best_val_flux_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training PIFNO3D for ALMA data")
    
    parser.add_argument('--tto_epochs', type=int,   default=10,   help='Epoche di TTO per campione')
    parser.add_argument('--tto_lr',     type=float, default=5e-6, help='Learning rate TTO (<<lr training)')
    parser.add_argument('--dataset_path', type=str, default='dataset/simulations', help='Path to the dataset directory')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset instead of ALMA dataset')
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
    parser.add_argument('--alpha', type=float, default=0.03, help='Weighting factor between L1 and MS-SSIM')
    parser.add_argument('--act', type=str, default='gelu', choices=['gelu', 'relu', 'tanh', 'leaky_relu'], help='Activation function')

    args = parser.parse_args()
    
    main(args)