import argparse
import copy
import csv
import os
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)

from dataset.ALMA_dataset import ALMADataset
from models.fno2d import FNO2d
from models.fno3d import FNO3d
from models.lno2d import LNO2d
from models.lno3d import LNO3d
from models.losses import CombinedLoss
from models.utils import set_seed
from models.CLEAN import hogbom_clean_batch


# ============================================================
# TIMER
# ============================================================

class Timer:
    def __init__(self, device):
        self.use_cuda = device.type == "cuda"

        if self.use_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_ev.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        """
        Returns elapsed time in milliseconds.
        """
        if self.use_cuda:
            self.end_ev.record()
            torch.cuda.synchronize()
            return self.start_ev.elapsed_time(self.end_ev)

        return (time.perf_counter() - self._t0) * 1000.0


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_all_parameters(model):
    """
    Count all parameters, including frozen parameters.
    """
    return sum(p.numel() for p in model.parameters())


def print_model_info(models_to_run):
    print("\n" + "=" * 100)
    print("MODEL INFORMATION")
    print("=" * 100)

    header = (
        f"{'Model':<18}"
        f"{'Parameters':>18}"
        f"{'Parameters (M)':>18}"
        f"{'Trainable':>18}"
    )

    print(header)
    print("-" * 100)

    for name, (model, _, _, _) in models_to_run.items():
        trainable = count_parameters(model)
        total = count_all_parameters(model)

        print(
            f"{name:<18}"
            f"{total:>18,}"
            f"{total / 1e6:>18.3f}"
            f"{trainable:>18,}"
        )

    print("=" * 100)


def get_parameter_statistics(models_to_run):
    """
    Returns a dictionary containing parameter counts.
    """
    stats = {}

    for name, (model, _, _, _) in models_to_run.items():
        total = count_all_parameters(model)
        trainable = count_parameters(model)

        stats[name] = {
            "parameters": total,
            "parameters_m": total / 1e6,
            "trainable_parameters": trainable,
            "trainable_parameters_m": trainable / 1e6,
        }

    return stats


# ============================================================
# MODEL LOADERS
# ============================================================

def load_lno2d(path, args, device):
    model = LNO2d(
        modes1=[args.modes_lno2d] * args.fourier_layers,
        modes2=[args.modes_lno2d] * args.fourier_layers,
        width=args.width_lno2d,
        in_dim=args.channels + 2,
        out_dim=args.channels,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_lno3d(path, args, device):
    model = LNO3d(
        modes1=[args.modes_z_lno3d] * args.fourier_layers,
        modes2=[args.modes_lno3d] * args.fourier_layers,
        modes3=[args.modes_lno3d] * args.fourier_layers,
        width=args.width_lno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_pilno2d(path, args, device):
    model = LNO2d(
        modes1=[args.modes_pilno2d] * args.fourier_layers,
        modes2=[args.modes_pilno2d] * args.fourier_layers,
        width=args.width_pilno2d,
        in_dim=args.channels + 2,
        out_dim=args.channels,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_pilno3d(path, args, device):
    model = LNO3d(
        modes1=[args.modes_z_pilno3d] * args.fourier_layers,
        modes2=[args.modes_pilno3d] * args.fourier_layers,
        modes3=[args.modes_pilno3d] * args.fourier_layers,
        width=args.width_pilno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_fno2d(path, args, device):
    model = FNO2d(
        modes1=[args.modes_fno2d] * args.fourier_layers,
        modes2=[args.modes_fno2d] * args.fourier_layers,
        width=args.width_fno2d,
        in_dim=args.channels + 2,
        out_dim=args.channels,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_fno3d(path, args, device):
    model = FNO3d(
        modes1=[args.modes_z_fno3d] * args.fourier_layers,
        modes2=[args.modes_fno3d] * args.fourier_layers,
        modes3=[args.modes_fno3d] * args.fourier_layers,
        width=args.width_fno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

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

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def load_pifno3d(path, args, device):
    model = FNO3d(
        modes1=[args.modes_z_pifno3d] * args.fourier_layers,
        modes2=[args.modes_pifno3d] * args.fourier_layers,
        modes3=[args.modes_pifno3d] * args.fourier_layers,
        width=args.width_pifno3d,
        in_dim=4,
        out_dim=1,
        pad_ratio=args.pad_ratio,
        act=args.act,
    ).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def infer_fno2d(model, dirty, device):
    out = model(dirty.to(device))
    return torch.clamp(out, min=0.0)


@torch.inference_mode()
def infer_fno3d(model, dirty, device):
    x = dirty.to(device).unsqueeze(1)

    out = model(x).squeeze(1)

    return torch.clamp(out, min=0.0)


INFER_FN = {
    "fno2d": infer_fno2d,
    "fno3d": infer_fno3d,
    "pifno2d": infer_fno2d,
    "pifno3d": infer_fno3d,
    "lno2d": infer_fno2d,
    "lno3d": infer_fno3d,
    "pilno2d": infer_fno2d,
    "pilno3d": infer_fno3d,
}


# ============================================================
# TEST-TIME OPTIMIZATION
# ============================================================

def tto_optimize(
    model,
    dirty,
    psf,
    device,
    channels,
    tto_epochs,
    tto_lr,
    is_3d,
):
    """
    Test-Time Optimization.

    The model is temporarily optimized on the current sample,
    then its original weights are restored.

    Returns:
        pred_tto
    """

    original_state = copy.deepcopy(model.state_dict())

    tto_criterion = CombinedLoss(
        lambda_data=0.0,
        lambda_phys=1.0,
        alpha=0.0,
        channels=channels,
    ).to(device)

    tto_opt = optim.Adam(
        model.parameters(),
        lr=tto_lr,
    )

    dirty_dev = dirty.to(device)
    psf_dev = psf.to(device)

    placeholder = torch.zeros_like(dirty_dev)

    model.train()

    for _ in range(tto_epochs):

        tto_opt.zero_grad()

        if is_3d:
            pred = model(
                dirty_dev.unsqueeze(1)
            ).squeeze(1)
        else:
            pred = model(dirty_dev)

        pred = torch.clamp(pred, min=0.0)

        loss, *_ = tto_criterion(
            pred,
            dirty_dev,
            placeholder,
            psf_dev,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        tto_opt.step()

    model.eval()

    with torch.inference_mode():

        if is_3d:
            pred_tto = model(
                dirty_dev.unsqueeze(1)
            ).squeeze(1)
        else:
            pred_tto = model(dirty_dev)

        pred_tto = torch.clamp(
            pred_tto,
            min=0.0,
        )

    model.load_state_dict(original_state)

    return pred_tto


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred, clean, device):
    """
    Compute quantitative reconstruction metrics.

    Metrics:
        Flux Error (%)
        Relative L2 Error (%)
        MAE
        PSNR (dB)
        SSIM
    """

    pred = pred.to(device)
    clean = clean.to(device)

    # --------------------------------------------------------
    # Remove accidental batch dimension
    # --------------------------------------------------------

    pred = pred.squeeze()
    clean = clean.squeeze()

    # --------------------------------------------------------
    # Normalize by target maximum
    # --------------------------------------------------------

    smax = clean.max()

    if smax <= 0:
        return None

    pred_norm = pred / smax
    clean_norm = clean / smax

    # --------------------------------------------------------
    # Flux error
    # --------------------------------------------------------

    mask = clean_norm > 1e-6

    true_flux = clean_norm[mask].sum()
    pred_flux = pred_norm[mask].sum()

    flux_err = (
        torch.abs(pred_flux - true_flux)
        / (true_flux + 1e-8)
        * 100.0
    )

    # --------------------------------------------------------
    # Relative L2 error
    # --------------------------------------------------------

    relative_l2 = (
        torch.linalg.vector_norm(
            pred_norm - clean_norm
        )
        / (
            torch.linalg.vector_norm(clean_norm)
            + 1e-8
        )
        * 100.0
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = torch.mean(
        torch.abs(pred_norm - clean_norm)
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    if pred_norm.ndim == 3:

        # [C, H, W]

        p_val = psnr(
            pred_norm.unsqueeze(0),
            clean_norm.unsqueeze(0),
            data_range=1.0,
        )

        s_val = ssim(
            pred_norm.unsqueeze(0),
            clean_norm.unsqueeze(0),
            data_range=1.0,
        )

    elif pred_norm.ndim == 2:

        # [H, W]

        p_val = psnr(
            pred_norm.unsqueeze(0).unsqueeze(0),
            clean_norm.unsqueeze(0).unsqueeze(0),
            data_range=1.0,
        )

        s_val = ssim(
            pred_norm.unsqueeze(0).unsqueeze(0),
            clean_norm.unsqueeze(0).unsqueeze(0),
            data_range=1.0,
        )

    else:

        raise ValueError(
            f"Unexpected tensor shape: {pred_norm.shape}"
        )

    return {
        "flux": flux_err.item(),
        "relative_l2": relative_l2.item(),
        "mae": mae.item(),
        "psnr": p_val.item(),
        "ssim": s_val.item(),
    }


def accumulate(acc, metrics):
    for key in acc:
        if key in metrics:
            acc[key].append(metrics[key])


# ============================================================
# VISUALIZATION
# ============================================================

def save_comparison_plot(
    sample_idx,
    dirty,
    clean,
    predictions,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    d = (
        dirty.cpu()
        .mean(dim=0)
        .numpy()
    )

    c = (
        clean.cpu()
        .mean(dim=0)
        .numpy()
    )

    c_max = (
        c.max()
        if c.max() > 0
        else 1.0
    )

    methods = list(predictions.keys())

    n_methods = len(methods)

    fig = plt.figure(
        figsize=(
            10,
            2.5 * (n_methods + 1),
        )
    )

    gs = gridspec.GridSpec(
        n_methods + 1,
        2,
        figure=fig,
        hspace=0.4,
        wspace=0.15,
    )

    # --------------------------------------------------------
    # Dirty
    # --------------------------------------------------------

    ax_d = fig.add_subplot(
        gs[0, 0]
    )

    ax_d.imshow(
        d,
        origin="lower",
        cmap="inferno",
    )

    ax_d.set_title(
        "Dirty (Input)",
        fontsize=9,
        fontweight="bold",
    )

    ax_d.axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ax_gt = fig.add_subplot(
        gs[0, 1]
    )

    ax_gt.imshow(
        c,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=c_max,
    )

    ax_gt.set_title(
        "Ground Truth",
        fontsize=9,
        fontweight="bold",
    )

    ax_gt.axis("off")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    for row, name in enumerate(
        methods,
        start=1,
    ):

        pred_np = (
            predictions[name]
            .cpu()
            .mean(dim=0)
            .numpy()
        )

        res_np = pred_np - c

        # Prediction

        ax_p = fig.add_subplot(
            gs[row, 0]
        )

        ax_p.imshow(
            pred_np,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=c_max,
        )

        ax_p.set_title(
            name,
            fontsize=8,
        )

        ax_p.axis("off")

        # Residual

        ax_r = fig.add_subplot(
            gs[row, 1]
        )

        lim = max(
            abs(res_np.min()),
            abs(res_np.max()),
            1e-9,
        )

        ax_r.imshow(
            res_np,
            origin="lower",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )

        ax_r.set_title(
            f"{name} — Residual",
            fontsize=8,
        )

        ax_r.axis("off")

    plt.suptitle(
        f"Methods Comparison — Sample {sample_idx}",
        fontsize=11,
        y=1.01,
    )

    path = os.path.join(
        output_dir,
        f"comparison_sample_{sample_idx:03d}.png",
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# RESULT AGGREGATION
# ============================================================

def aggregate_metrics(acc):
    """
    Compute mean/std for every metric.
    """

    avg_metrics = {}

    for method, values in acc.items():

        avg_metrics[method] = {}

        for key, value_list in values.items():

            if len(value_list) == 0:
                avg_metrics[method][key] = np.nan
                avg_metrics[method][f"{key}_std"] = np.nan
                continue

            arr = np.asarray(
                value_list,
                dtype=np.float64,
            )

            avg_metrics[method][key] = arr.mean()
            avg_metrics[method][f"{key}_std"] = arr.std()

    return avg_metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics_table(
    all_metrics,
    parameter_stats,
):
    print("\n" + "=" * 150)
    print("BENCHMARK RESULTS")
    print("=" * 150)

    header = (
        f"{'Method':<16}"
        f"{'Params (M)':>14}"
        f"{'Flux (%)':>18}"
        f"{'Rel. L2 (%)':>18}"
        f"{'MAE':>16}"
        f"{'PSNR (dB)':>18}"
        f"{'SSIM':>18}"
        f"{'Inference (ms)':>20}"
    )

    print(header)
    print("-" * 150)

    for name, metrics in all_metrics.items():

        if name in parameter_stats:
            params_m = parameter_stats[name][
                "parameters_m"
            ]
        else:
            params_m = 0.0

        def fmt(key):
            if np.isnan(metrics[key]):
                return "N/A"

            return (
                f"{metrics[key]:.4f} "
                f"± {metrics[key + '_std']:.4f}"
            )

        print(
            f"{name:<16}"
            f"{params_m:>14.3f}"
            f"{fmt('flux'):>18}"
            f"{fmt('relative_l2'):>18}"
            f"{fmt('mae'):>16}"
            f"{fmt('psnr'):>18}"
            f"{fmt('ssim'):>18}"
            f"{fmt('time_ms'):>20}"
        )

    print("=" * 150)

    print("\nTTO RESULTS")
    print("=" * 100)

    header = (
        f"{'Method':<20}"
        f"{'TTO Time (ms)':>25}"
        f"{'Total Time (ms)':>25}"
    )

    print(header)
    print("-" * 100)

    for name, metrics in all_metrics.items():

        if "tto_time_ms" not in metrics:
            continue

        if np.isnan(metrics["tto_time_ms"]):
            continue

        print(
            f"{name:<20}"
            f"{metrics['tto_time_ms']:>25.4f}"
            f"{metrics['tto_time_ms_std']:>25.4f}"
        )

    print()


# ============================================================
# METRICS CSV
# ============================================================

def save_metrics_csv(
    all_metrics,
    parameter_stats,
    output_dir,
):
    path = os.path.join(
        output_dir,
        "benchmark_results.csv",
    )

    rows = []

    for name, metrics in all_metrics.items():

        row = {
            "method": name,
            "parameters": parameter_stats.get(
                name,
                {}
            ).get(
                "parameters",
                0,
            ),
            "parameters_m": parameter_stats.get(
                name,
                {}
            ).get(
                "parameters_m",
                0.0,
            ),
        }

        for key, value in metrics.items():
            row[key] = value

        rows.append(row)

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Benchmark CSV salvato in: {path}"
    )


def save_parameter_csv(
    parameter_stats,
    output_dir,
):
    path = os.path.join(
        output_dir,
        "model_parameters.csv",
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "method",
            "parameters",
            "parameters_m",
            "trainable_parameters",
            "trainable_parameters_m",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for name, stats in parameter_stats.items():

            writer.writerow({
                "method": name,
                **stats,
            })

    print(
        f"Parameter CSV salvato in: {path}"
    )


# ============================================================
# METRICS CHART
# ============================================================

def save_metrics_chart(
    all_metrics,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    methods = list(
        all_metrics.keys()
    )

    x = np.arange(
        len(methods)
    )

    metric_keys = [
        "flux",
        "relative_l2",
        "psnr",
        "ssim",
    ]

    metric_labels = [
        "Flux Error (%) ↓",
        "Relative L2 Error (%) ↓",
        "PSNR (dB) ↑",
        "SSIM ↑",
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 6),
    )

    for ax, key, label in zip(
        axes,
        metric_keys,
        metric_labels,
    ):

        means = np.array([
            all_metrics[m][key]
            for m in methods
        ])

        stds = np.array([
            all_metrics[m][
                f"{key}_std"
            ]
            for m in methods
        ])

        ax.bar(
            x,
            means,
            0.55,
            yerr=stds,
            capsize=4,
            alpha=0.82,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            methods,
            rotation=30,
            ha="right",
            fontsize=8,
        )

        ax.set_title(
            label,
            fontsize=10,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

    plt.suptitle(
        "Reconstruction Metrics",
        fontsize=12,
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "metrics_summary.png",
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Metrics chart salvato in: {path}"
    )


# ============================================================
# PARAMETER CHART
# ============================================================

def save_parameter_chart(
    parameter_stats,
    output_dir,
):
    methods = list(
        parameter_stats.keys()
    )

    values = [
        parameter_stats[m][
            "parameters_m"
        ]
        for m in methods
    ]

    x = np.arange(
        len(methods)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    bars = ax.bar(
        x,
        values,
        width=0.6,
        alpha=0.82,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        methods,
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Trainable Parameters (Millions)"
    )

    ax.set_title(
        "Model Complexity"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f} M",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "model_parameters.png",
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Parameter chart salvato in: {path}"
    )


# ============================================================
# TIME CHART
# ============================================================

def save_time_chart(
    all_metrics,
    output_dir,
):
    methods = list(
        all_metrics.keys()
    )

    inference = np.array([
        all_metrics[m]["time_ms"]
        for m in methods
    ])

    tto = np.array([
        all_metrics[m].get(
            "tto_time_ms",
            0.0,
        )
        for m in methods
    ])

    x = np.arange(
        len(methods)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        x,
        inference,
        width=0.6,
        label="Inference",
    )

    ax.bar(
        x,
        tto,
        width=0.6,
        bottom=inference,
        label="TTO",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        methods,
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Time per sample (ms)"
    )

    ax.set_title(
        "Inference and TTO Cost"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "time_summary.png",
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Time chart salvato in: {path}"
    )


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(args):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(42)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    full_dataset = ALMADataset(
        args.dataset_path
    )

    total = len(
        full_dataset
    )

    tr_size = int(
        0.70 * total
    )

    val_size = int(
        0.15 * total
    )

    te_size = (
        total
        - tr_size
        - val_size
    )

    _, _, test_dataset = random_split(
        full_dataset,
        [
            tr_size,
            val_size,
            te_size,
        ],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    print(
        f"Dataset size: {total}"
    )

    print(
        f"Test set: {te_size} samples\n"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models_to_run = {}

    checkpoint_map = {

        "FNO2d": (
            args.fno2d,
            load_fno2d,
            False,
            False,
            None,
        ),

        "FNO3d": (
            args.fno3d,
            load_fno3d,
            True,
            False,
            None,
        ),

        "PI-FNO2d": (
            args.pifno2d,
            load_pifno2d,
            False,
            True,
            (
                args.tto_epochs_pifno2d,
                args.tto_lr_pifno2d,
            ),
        ),

        "PI-FNO3d": (
            args.pifno3d,
            load_pifno3d,
            True,
            True,
            (
                args.tto_epochs_pifno3d,
                args.tto_lr_pifno3d,
            ),
        ),

        "LNO2d": (
            args.lno2d,
            load_lno2d,
            False,
            False,
            None,
        ),

        "LNO3d": (
            args.lno3d,
            load_lno3d,
            True,
            False,
            None,
        ),

        "PI-LNO2d": (
            args.pilno2d,
            load_pilno2d,
            False,
            True,
            (
                args.tto_epochs_pilno2d,
                args.tto_lr_pilno2d,
            ),
        ),

        "PI-LNO3d": (
            args.pilno3d,
            load_pilno3d,
            True,
            True,
            (
                args.tto_epochs_pilno3d,
                args.tto_lr_pilno3d,
            ),
        ),
    }

    for (
        name,
        (
            path,
            loader_fn,
            is_3d,
            do_tto,
            tto_cfg,
        ),
    ) in checkpoint_map.items():

        if path and os.path.isfile(path):

            print(
                f"Loading {name} from {path}..."
            )

            model = loader_fn(
                path,
                args,
                device,
            )

            models_to_run[name] = (
                model,
                is_3d,
                do_tto,
                tto_cfg,
            )

        elif path:

            print(
                f"[WARN] Checkpoint not found "
                f"for {name}: {path} — skipped"
            )

    if not models_to_run:

        raise RuntimeError(
            "No valid checkpoint found. "
            "Please check the paths."
        )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print_model_info(
        models_to_run
    )

    parameter_stats = (
        get_parameter_statistics(
            models_to_run
        )
    )

    save_parameter_csv(
        parameter_stats,
        args.output_dir,
    )

    save_parameter_chart(
        parameter_stats,
        args.output_dir,
    )

    # --------------------------------------------------------
    # Method names
    # --------------------------------------------------------

    method_names = []

    for (
        name,
        (
            _,
            _,
            do_tto,
            _,
        ),
    ) in models_to_run.items():

        method_names.append(name)

        if do_tto:
            method_names.append(
                f"{name}+TTO"
            )

    method_names.append(
        "CLEAN"
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    metric_keys = [
        "flux",
        "relative_l2",
        "mae",
        "psnr",
        "ssim",
        "time_ms",
        "tto_time_ms",
    ]

    acc = {
        method: {
            key: []
            for key in metric_keys
        }
        for method in method_names
    }

    n_valid = {
        method: 0
        for method in method_names
    }

    timer = Timer(device)

    # --------------------------------------------------------
    # Main benchmark loop
    # --------------------------------------------------------

    for sample_idx, (
        dirty,
        clean,
        psf,
    ) in enumerate(test_loader):

        dirty_s = dirty[0]
        clean_s = clean[0]

        predictions_for_plot = {}

        # ====================================================
        # NEURAL MODELS
        # ====================================================

        for name, (
            model,
            is_3d,
            do_tto,
            tto_cfg,
        ) in models_to_run.items():

            infer_fn = (
                infer_fno3d
                if is_3d
                else infer_fno2d
            )

            # ------------------------------------------------
            # Standard inference
            # ------------------------------------------------

            timer.start()

            pred_pre = infer_fn(
                model,
                dirty,
                device,
            )[0]

            t_pre = timer.stop()

            metrics_pre = compute_metrics(
                pred_pre,
                clean_s,
                device,
            )

            if metrics_pre is not None:

                metrics_pre[
                    "time_ms"
                ] = t_pre

                accumulate(
                    acc[name],
                    metrics_pre,
                )

                n_valid[name] += 1

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            if sample_idx < args.n_viz:

                predictions_for_plot[
                    name
                ] = (
                    pred_pre
                    .detach()
                    .cpu()
                )

            # ------------------------------------------------
            # TTO
            # ------------------------------------------------

            if do_tto:

                tto_epochs, tto_lr = (
                    tto_cfg
                )

                timer.start()

                pred_tto = tto_optimize(
                    model,
                    dirty,
                    psf,
                    device,
                    channels=args.channels,
                    tto_epochs=tto_epochs,
                    tto_lr=tto_lr,
                    is_3d=is_3d,
                )[0]

                t_tto = timer.stop()

                metrics_tto = compute_metrics(
                    pred_tto,
                    clean_s,
                    device,
                )

                tto_name = (
                    f"{name}+TTO"
                )

                if metrics_tto is not None:

                    metrics_tto[
                        "time_ms"
                    ] = t_pre

                    metrics_tto[
                        "tto_time_ms"
                    ] = t_tto

                    accumulate(
                        acc[tto_name],
                        metrics_tto,
                    )

                    n_valid[tto_name] += 1

                if sample_idx < args.n_viz:

                    predictions_for_plot[
                        tto_name
                    ] = (
                        pred_tto
                        .detach()
                        .cpu()
                    )

        # ====================================================
        # HOGBOM CLEAN
        # ====================================================

        timer.start()

        pred_clean = (
            hogbom_clean_batch(
                dirty.to(device),
                psf.to(device),
                n_iter=1000,
            )[0]
        )

        t_clean = timer.stop()

        metrics_clean = compute_metrics(
            pred_clean,
            clean_s,
            device,
        )

        if metrics_clean is not None:

            metrics_clean[
                "time_ms"
            ] = t_clean

            accumulate(
                acc["CLEAN"],
                metrics_clean,
            )

            n_valid["CLEAN"] += 1

        if sample_idx < args.n_viz:

            predictions_for_plot[
                "CLEAN"
            ] = (
                pred_clean
                .detach()
                .cpu()
            )

            save_comparison_plot(
                sample_idx,
                dirty_s,
                clean_s,
                predictions_for_plot,
                output_dir=os.path.join(
                    args.output_dir,
                    "comparisons",
                ),
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"  [{sample_idx + 1:>4}/{te_size}]",
            end="\r",
        )

    print("\n")

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    avg_metrics = aggregate_metrics(
        acc
    )

    # --------------------------------------------------------
    # Add total TTO time
    # --------------------------------------------------------

    for name in avg_metrics:

        if "tto_time_ms" in avg_metrics[name]:

            inference_mean = (
                avg_metrics[name][
                    "time_ms"
                ]
            )

            tto_mean = (
                avg_metrics[name][
                    "tto_time_ms"
                ]
            )

            avg_metrics[name][
                "total_time_ms"
            ] = (
                inference_mean
                + tto_mean
            )

            avg_metrics[name][
                "total_time_ms_std"
            ] = np.sqrt(
                avg_metrics[name][
                    "time_ms_std"
                ] ** 2
                +
                avg_metrics[name][
                    "tto_time_ms_std"
                ] ** 2
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_metrics_table(
        avg_metrics,
        parameter_stats,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_metrics_csv(
        avg_metrics,
        parameter_stats,
        args.output_dir,
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    save_metrics_chart(
        avg_metrics,
        args.output_dir,
    )

    save_time_chart(
        avg_metrics,
        args.output_dir,
    )

    print(
        "\nVisualizations saved in: "
        f"{os.path.join(args.output_dir, 'comparisons')}"
    )

    print(
        "\nBenchmark completed."
    )


# ============================================================
# ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FNO2d/3d, LNO2d/3d, "
            "PI variants and CLEAN."
        )
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    parser.add_argument(
        "--fno2d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/fno2d.pth",
        help="Path checkpoint FNO2d",
    )

    parser.add_argument(
        "--fno3d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/fno3d.pth",
        help="Path checkpoint FNO3d",
    )

    parser.add_argument(
        "--pifno2d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/pifno2d.pth",
        help="Path checkpoint PI-FNO2d",
    )

    parser.add_argument(
        "--pifno3d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/pifno3d.pth",
        help="Path checkpoint PI-FNO3d",
    )

    parser.add_argument(
        "--lno2d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/lno2d.pth",
        help="Path checkpoint LNO2d",
    )

    parser.add_argument(
        "--lno3d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/lno3d.pth",
        help="Path checkpoint LNO3d",
    )

    parser.add_argument(
        "--pilno2d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/pilno2d.pth",
        help="Path checkpoint PI-LNO2d",
    )

    parser.add_argument(
        "--pilno3d",
        type=str,
        default="/data1/rtessitore/ALMA-PINO/checkpoints/pilno3d.pth",
        help="Path checkpoint PI-LNO3d",
    )

    # --------------------------------------------------------
    # LNO2D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_lno2d",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width_lno2d",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # LNO3D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_lno3d",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--modes_z_lno3d",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width_lno3d",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # PI-LNO2D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_pilno2d",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--width_pilno2d",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # PI-LNO3D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_pilno3d",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--modes_z_pilno3d",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width_pilno3d",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # FNO2D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_fno2d",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--width_fno2d",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # FNO3D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_fno3d",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--modes_z_fno3d",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width_fno3d",
        type=int,
        default=64,
    )

    # --------------------------------------------------------
    # PI-FNO2D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_pifno2d",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--width_pifno2d",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # PI-FNO3D
    # --------------------------------------------------------

    parser.add_argument(
        "--modes_pifno3d",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--modes_z_pifno3d",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width_pifno3d",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # General model parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--fourier_layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--act",
        type=str,
        default="gelu",
        choices=[
            "gelu",
            "relu",
            "tanh",
            "leaky_relu",
        ],
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data1/rtessitore/almasim/alma_dataset/dataset",
    )

    # --------------------------------------------------------
    # TTO
    # --------------------------------------------------------

    parser.add_argument(
        "--tto_epochs_pifno2d",
        type=int,
        default=10,
        help="TTO epochs for PI-FNO2d",
    )

    parser.add_argument(
        "--tto_lr_pifno2d",
        type=float,
        default=5e-6,
        help="TTO learning rate for PI-FNO2d",
    )

    parser.add_argument(
        "--tto_epochs_pifno3d",
        type=int,
        default=10,
        help="TTO epochs for PI-FNO3d",
    )

    parser.add_argument(
        "--tto_lr_pifno3d",
        type=float,
        default=1e-7,
        help="TTO learning rate for PI-FNO3d",
    )

    # --------------------------------------------------------
    # PI-LNO2D TTO
    # --------------------------------------------------------

    parser.add_argument(
        "--tto_epochs_pilno2d",
        type=int,
        default=10,
        help="TTO epochs for PI-LNO2d",
    )

    parser.add_argument(
        "--tto_lr_pilno2d",
        type=float,
        default=5e-6,
        help="TTO learning rate for PI-LNO2d",
    )

    # --------------------------------------------------------
    # PI-LNO3D TTO
    # --------------------------------------------------------

    parser.add_argument(
        "--tto_epochs_pilno3d",
        type=int,
        default=10,
        help="TTO epochs for PI-LNO3d",
    )

    parser.add_argument(
        "--tto_lr_pilno3d",
        type=float,
        default=1e-7,
        help="TTO learning rate for PI-LNO3d",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_benchmark",
    )

    parser.add_argument(
        "--n_viz",
        type=int,
        default=5,
        help=(
            "Number of samples for which "
            "to save comparative plots"
        ),
    )

    args = parser.parse_args()

    run_benchmark(args)

