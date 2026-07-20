import argparse
import copy
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from dataset.ALMA_dataset import ALMADataset
from models.fno2d import FNO2d
from models.fno3d import FNO3d
from models.losses import CombinedLoss
from models.utils import set_seed
from models.CLEAN import hogbom_clean_batch

class Timer:
    def __init__(self, device):
        self.use_cuda = device.type == "cuda"
        if self.use_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev   = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_ev.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Ritorna il tempo trascorso in millisecondi."""
        if self.use_cuda:
            self.end_ev.record()
            torch.cuda.synchronize()
            return self.start_ev.elapsed_time(self.end_ev)   # ms
        else:
            return (time.perf_counter() - self._t0) * 1000   # ms



def load_fno2d(path, args, device):
    model = FNO2d(
        modes1=[args.modes_fno2d] * args.fourier_layers,
        modes2=[args.modes_fno2d] * args.fourier_layers,
        width=args.width_fno2d,
        in_dim=args.channels + 2,
        out_dim=args.channels,
        act=args.act,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_fno3d(path, args, device):
    model = FNO3d(
        modes1=[args.modes_z_fno3d] * args.fourier_layers,
        modes2=[args.modes_fno3d]   * args.fourier_layers,
        modes3=[args.modes_fno3d]   * args.fourier_layers,
        width=args.width_fno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_pifno2d(path, args, device):
    model = FNO2d(
        modes1=[args.modes_pifno2d] * args.fourier_layers,
        modes2=[args.modes_pifno2d] * args.fourier_layers,
        width=args.width_pifno2d,
        in_dim=args.channels + 2,
        out_dim=args.channels,
        act=args.act,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_pifno3d(path, args, device):
    model = FNO3d(
        modes1=[args.modes_z_pifno3d] * args.fourier_layers,
        modes2=[args.modes_pifno3d]   * args.fourier_layers,
        modes3=[args.modes_pifno3d]   * args.fourier_layers,
        width=args.width_pifno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def infer_fno2d(model, dirty, device):
    return torch.clamp(model(dirty.to(device)), min=0.0)


@torch.no_grad()
def infer_fno3d(model, dirty, device):
    x = dirty.to(device).unsqueeze(1)           # [B, 1, C, H, W]
    out = model(x).squeeze(1)                   # [B, C, H, W]
    return torch.clamp(out, min=0.0)


INFER_FN = {
    "fno2d":   infer_fno2d,
    "fno3d":   infer_fno3d,
    "pifno2d": infer_fno2d,
    "pifno3d": infer_fno3d,
}


def tto_optimize(model, dirty, psf, device, channels, tto_epochs, tto_lr, is_3d):
    original_state = copy.deepcopy(model.state_dict())

    tto_criterion = CombinedLoss(
        lambda_data=0.0,
        lambda_phys=1.0,
        alpha=0.0,
        channels=channels,
    ).to(device)

    tto_opt     = optim.Adam(model.parameters(), lr=tto_lr)
    dirty_dev   = dirty.to(device)
    psf_dev     = psf.to(device)
    placeholder = torch.zeros_like(dirty_dev)  

    model.train()
    for _ in range(tto_epochs):
        tto_opt.zero_grad()

        if is_3d:
            pred = torch.clamp(model(dirty_dev.unsqueeze(1)).squeeze(1), min=0.0)
        else:
            pred = torch.clamp(model(dirty_dev), min=0.0)

        loss, *_ = tto_criterion(pred, dirty_dev, placeholder, psf_dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tto_opt.step()

    model.eval()
    with torch.no_grad():
        if is_3d:
            pred_tto = torch.clamp(model(dirty_dev.unsqueeze(1)).squeeze(1), min=0.0)
        else:
            pred_tto = torch.clamp(model(dirty_dev), min=0.0)

    model.load_state_dict(original_state)
    return pred_tto



def compute_metrics(pred, clean, device):
    pred  = pred.to(device)
    clean = clean.to(device)
    smax  = clean.max()

    if smax <= 0:
        return None

    mask      = clean > (0.01 * smax)
    true_flux = clean[mask].sum()
    pred_flux = pred[mask].sum()
    flux_err  = torch.abs(pred_flux - true_flux) / (true_flux + 1e-8) * 100

    p_val = psnr(pred.unsqueeze(0), clean.unsqueeze(0), data_range=smax.item())
    s_val = ssim(pred.unsqueeze(0), clean.unsqueeze(0), data_range=smax.item())

    return {
        "flux": flux_err.item(),
        "psnr": p_val.item(),
        "ssim": s_val.item(),
    }


def accumulate(acc, m):
    for k in acc:
        acc[k] += m[k]



def save_comparison_plot(sample_idx, dirty, clean, predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    d = dirty.cpu().mean(dim=0).numpy()
    c = clean.cpu().mean(dim=0).numpy()
    c_max = c.max() if c.max() > 0 else 1.0

    methods  = list(predictions.keys())
    n_methods = len(methods)

    fig = plt.figure(figsize=(10, 2.5 * (n_methods + 1)))
    gs  = gridspec.GridSpec(
        n_methods + 1, 2,
        figure=fig,
        hspace=0.4, wspace=0.15,
    )

    ax_d = fig.add_subplot(gs[0, 0])
    ax_d.imshow(d, origin="lower", cmap="inferno")
    ax_d.set_title("Dirty (Input)", fontsize=9, fontweight="bold")
    ax_d.axis("off")

    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.imshow(c, origin="lower", cmap="inferno")
    ax_gt.set_title("Ground Truth", fontsize=9, fontweight="bold")
    ax_gt.axis("off")

    for row, name in enumerate(methods, start=1):
        pred_np = predictions[name].cpu().mean(dim=0).numpy()
        res_np  = pred_np - c

        ax_p = fig.add_subplot(gs[row, 0])
        ax_p.imshow(pred_np, origin="lower", cmap="inferno",
                    vmin=0, vmax=c_max)
        ax_p.set_title(f"{name}", fontsize=8)
        ax_p.axis("off")

        ax_r = fig.add_subplot(gs[row, 1])
        lim = max(abs(res_np.min()), abs(res_np.max()), 1e-9)
        ax_r.imshow(res_np, origin="lower", cmap="RdBu_r",
                    vmin=-lim, vmax=lim)
        ax_r.set_title(f"{name} — Residual", fontsize=8)
        ax_r.axis("off")

    plt.suptitle(f"Methods Comparison — Sample {sample_idx}", fontsize=11, y=1.01)
    path = os.path.join(output_dir, f"comparison_sample_{sample_idx:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_metrics_chart(all_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    methods = list(all_metrics.keys())
    n       = len(methods)
    x       = np.arange(n)
    width   = 0.55

    metric_keys   = ["flux", "psnr", "ssim", "time_ms"]
    metric_labels = ["Flux Error (%) ↓", "PSNR (dB) ↑", "SSIM ↑", "Time per Sample (ms) ↓"]
    colors        = ["#e07b54", "#4e9ab3", "#6abf7b", "#b07cc6"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, key, label, color in zip(axes, metric_keys, metric_labels, colors):
        values = [all_metrics[m][key] for m in methods]
        bars   = ax.bar(x, values, width, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Metrics Summary", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Metrics chart salvato in: {path}")


def print_metrics_table(all_metrics):
    header = f"\n{'Metodo':<20} {'Flux Error (%)':>16} {'PSNR (dB)':>12} {'SSIM':>10} {'Time (ms)':>12}"
    print(header)
    print("─" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['flux']:>16.4f} {m['psnr']:>12.4f} "
              f"{m['ssim']:>10.4f} {m['time_ms']:>12.2f}")
    print()


def run_benchmark(args):
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = ALMADataset(args.dataset_path)
    total    = len(full_dataset)
    tr_size  = int(0.7 * total)
    val_size = int(0.15 * total)
    te_size  = total - tr_size - val_size

    _, _, test_dataset = random_split(
        full_dataset,
        [tr_size, val_size, te_size],
        generator=torch.Generator().manual_seed(42),  
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test set: {te_size} samples\n")

    #Load models
    models_to_run = {}     # { name: (model, is_3d) }
    checkpoint_map = {
        "FNO2d":    (args.fno2d,   load_fno2d,   False, False, None),
        "FNO3d":    (args.fno3d,   load_fno3d,   True,  False, None),
        "PI-FNO2d": (args.pifno2d, load_pifno2d, False, True,
                     (args.tto_epochs_pifno2d, args.tto_lr_pifno2d)),
        "PI-FNO3d": (args.pifno3d, load_pifno3d, True,  True,
                     (args.tto_epochs_pifno3d, args.tto_lr_pifno3d)),
    }

    for name, (path, loader_fn, is_3d, do_tto, tto_cfg) in checkpoint_map.items():
        if path and os.path.isfile(path):
            print(f"Loading {name} from {path}...")
            models_to_run[name] = (loader_fn(path, args, device), is_3d, do_tto, tto_cfg)
        elif path:
            print(f"[WARN] Checkpoint not found for {name}: {path} — skipped")

    if not models_to_run:
        raise RuntimeError("No valid checkpoint found. Please check the paths.")

    method_names = []
    for name, (_, _, do_tto, _) in models_to_run.items():
        method_names.append(name)
        if do_tto:
            method_names.append(f"{name}+TTO")
    method_names += ["CLEAN"]

    acc = {m: {"flux": 0.0, "psnr": 0.0, "ssim": 0.0, "time_ms": 0.0} for m in method_names}
    n_valid = {m: 0 for m in method_names}
    timer = Timer(device)

    for sample_idx, (dirty, clean, psf) in enumerate(test_loader):
        # dirty, clean, psf: [1, C, H, W]
        dirty_s = dirty[0]   # [C, H, W] 
        clean_s = clean[0]

        predictions_for_plot = {}   

        for name, (model, is_3d, do_tto, tto_cfg) in models_to_run.items():
            infer_fn = infer_fno2d if not is_3d else infer_fno3d

            #Pre-TTO
            timer.start()
            pred_pre = infer_fn(model, dirty, device)[0]   # [C, H, W]
            t_pre = timer.stop()

            m_pre = compute_metrics(pred_pre, clean_s, device)
            if m_pre:
                m_pre["time_ms"] = t_pre
                accumulate(acc[name], m_pre)
                n_valid[name] += 1

            #Post-TTO 
            if sample_idx < args.n_viz:
                predictions_for_plot[name] = pred_pre.detach().cpu()

            if do_tto:
                tto_epochs, tto_lr = tto_cfg
                timer.start()
                pred_tto = tto_optimize(
                    model, dirty, psf, device,
                    channels=args.channels,
                    tto_epochs=tto_epochs,
                    tto_lr=tto_lr,
                    is_3d=is_3d,
                )[0]   # [C, H, W]
                t_tto = timer.stop()

                m_tto = compute_metrics(pred_tto, clean_s, device)
                if m_tto:
                    m_tto["time_ms"] = t_tto
                    accumulate(acc[f"{name}+TTO"], m_tto)
                    n_valid[f"{name}+TTO"] += 1

                if sample_idx < args.n_viz:
                    predictions_for_plot[f"{name}+TTO"] = pred_tto.detach().cpu()

        # CLEAN
        timer.start()
        pred_clean = hogbom_clean_batch(dirty.to(device), psf.to(device), n_iter=1000)[0]
        t_clean = timer.stop()

        m_clean = compute_metrics(pred_clean, clean_s, device)
        if m_clean:
            m_clean["time_ms"] = t_clean
            accumulate(acc["CLEAN"], m_clean)
            n_valid["CLEAN"] += 1

        if sample_idx < args.n_viz:
            predictions_for_plot["CLEAN"] = pred_clean.detach().cpu()
            save_comparison_plot(
                sample_idx,
                dirty_s,
                clean_s,
                predictions_for_plot,
                output_dir=os.path.join(args.output_dir, "comparisons"),
            )

        # Progress
        print(f"  [{sample_idx + 1:>4}/{te_size}]", end="\r")

    avg_metrics = {}
    for m in method_names:
        n = n_valid[m]
        if n > 0:
            avg_metrics[m] = {k: v / n for k, v in acc[m].items()}

    print_metrics_table(avg_metrics)
    save_metrics_chart(avg_metrics, args.output_dir)
    print(f"\nVisualizations saved in: {os.path.join(args.output_dir, 'comparisons')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FNO2d/3d/PI (pre+post TTO) vs CLEAN")

    parser.add_argument("--fno2d",   type=str, default=None, help="Path checkpoint FNO2d")
    parser.add_argument("--fno3d",   type=str, default=None, help="Path checkpoint FNO3d")
    parser.add_argument("--pifno2d", type=str, default=None, help="Path checkpoint PI-FNO2d")
    parser.add_argument("--pifno3d", type=str, default=None, help="Path checkpoint PI-FNO3d")

    # FNO2D
    parser.add_argument("--modes_fno2d", type=int,   default=8)
    parser.add_argument("--width_fno2d", type=int,   default=32)

    # FNO3D
    parser.add_argument("--modes_fno3d", type=int,   default=12)
    parser.add_argument("--modes_z_fno3d", type=int, default=4)
    parser.add_argument("--width_fno3d", type=int,   default=64)
  

    # PI-FNO2D
    parser.add_argument("--modes_pifno2d", type=int,   default=8)
    parser.add_argument("--width_pifno2d", type=int,   default=32)

    # PI-FNO3D
    parser.add_argument("--modes_pifno3d", type=int,   default=12)
    parser.add_argument("--modes_z_pifno3d", type=int, default=8)
    parser.add_argument("--width_pifno3d", type=int,   default=64)


    parser.add_argument("--fourier_layers", type=int,   default=4)
    parser.add_argument("--channels",       type=int,   default=16)
    parser.add_argument("--pad_ratio",      type=float, default=0.1)
    parser.add_argument("--act",            type=str,   default="gelu",
                        choices=["gelu", "relu", "tanh", "leaky_relu"])

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="dataset/simulations")

    # TTO
    parser.add_argument("--tto_epochs_pifno2d", type=int,   default=10,
                        help="TTO epochs for PI-FNO2d")
    parser.add_argument("--tto_lr_pifno2d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-FNO2d (Keep << lr training)")
    parser.add_argument("--tto_epochs_pifno3d", type=int,   default=5,
                        help="TTO epochs for PI-FNO3d")
    parser.add_argument("--tto_lr_pifno3d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-FNO3d (Keep << lr training)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_benchmark")
    parser.add_argument("--n_viz",      type=int, default=5,
                        help="Number of samples for which to save comparative plots")

    args = parser.parse_args()
    run_benchmark(args)