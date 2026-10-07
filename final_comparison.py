import argparse
import copy
import os
import sys
import time

import matplotlib
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
from models.lno2d import LNO2d
from models.lno3d import LNO3d
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
    "fno2d": infer_fno2d,
    "fno3d": infer_fno3d,
    "pifno2d": infer_fno2d,
    "pifno3d": infer_fno3d,
    "lno2d": infer_fno2d,
    "lno3d": infer_fno3d,
    "pilno2d": infer_fno2d,
    "pilno3d": infer_fno3d,
}

def uv_to_psf(uv_mask):
    uv_shifted = torch.fft.ifftshift(uv_mask, dim=(-2, -1))
    psf_complex = torch.fft.ifft2(uv_shifted, dim=(-2, -1))
    psf_spatial = torch.fft.fftshift(psf_complex.real, dim=(-2, -1))
    psf_max = psf_spatial.amax(dim=(-2, -1), keepdim=True)
    psf = psf_spatial / (psf_max + 1e-8)

    return psf

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

def estimate_sigma_mad_clipping(
    image,
    n_iter=5,
    clip_sigma=3.0,
):
    """
    Estimate the noise standard deviation using iterative
    MAD-based sigma clipping.

    Parameters
    ----------
    image : torch.Tensor
        2D image [H, W].
    n_iter : int
        Number of clipping iterations.
    clip_sigma : float
        Clipping threshold in units of sigma.

    Returns
    -------
    sigma : torch.Tensor
        Robust estimate of the noise standard deviation.
    """

    x = image.flatten()

    # Initial robust estimate
    median = torch.median(x)
    mad = torch.median(torch.abs(x - median))

    sigma = 1.4826 * mad

    # Iterative sigma clipping
    for _ in range(n_iter):

        mask = torch.abs(x - median) < clip_sigma * sigma
        x_clipped = x[mask]

        if x_clipped.numel() == 0:
            break

        median = torch.median(x_clipped)
        mad = torch.median(torch.abs(x_clipped - median))

        sigma = 1.4826 * mad

    return sigma

def estimate_cube_sigma(dirty, n_iter=5, clip_sigma=3.0):
    """
    Estimate one noise sigma per spectral channel.

    dirty: [Z, H, W]
    returns: [Z]
    """

    sigmas = []

    for z in range(dirty.shape[0]):
        sigma_z = estimate_sigma_mad_clipping(
            dirty[z],
            n_iter=n_iter,
            clip_sigma=clip_sigma,
        )
        sigmas.append(sigma_z)

    return torch.stack(sigmas)

def compute_metrics(pred, clean, dirty, device, sigma_threshold=5.0):

    pred = pred.to(device)
    clean = clean.to(device)
    dirty = dirty.to(device)

    # --------------------------------------------------
    # Noise estimation: one sigma per spectral channel
    # --------------------------------------------------

    sigmas = estimate_cube_sigma(dirty)

    # --------------------------------------------------
    # Source mask: clean > k sigma
    # --------------------------------------------------

    threshold = sigma_threshold * sigmas[:, None, None]

    mask = clean > threshold

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    smax = clean.max()

    if smax <= 0:
        return None

    pred_norm = pred / smax
    clean_norm = clean / smax

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    true_flux = clean_norm[mask].sum()
    pred_flux = pred_norm[mask].sum()

    flux_err = (
        torch.abs(pred_flux - true_flux)
        / (true_flux + 1e-8)
        * 100
    )

    source_mae = torch.abs(
        pred_norm[mask] - clean_norm[mask]
    ).mean()

    p_val = psnr(
        pred_norm.unsqueeze(0),
        clean_norm.unsqueeze(0),
        data_range=1.0
    )

    s_val = ssim(
        pred_norm.unsqueeze(0),
        clean_norm.unsqueeze(0),
        data_range=1.0
    )

    return {
        "flux": flux_err.item(),
        "mae_src": source_mae.item(),
        "psnr": p_val.item(),
        "ssim": s_val.item(),
    }



def compute_metrics_1(pred, clean, device):
    pred  = pred.to(device)
    clean = clean.to(device)
    smax  = clean.max()

    if smax <= 0:
        return None

    pred_norm  = pred  / smax
    clean_norm = clean / smax

    mask      = clean_norm > 0.01
    true_flux = clean_norm[mask].sum()
    pred_flux = pred_norm[mask].sum()
    flux_err  = torch.abs(pred_flux - true_flux) / (true_flux + 1e-8) * 100

    source_mae = torch.abs(
        pred_norm[mask] - clean_norm[mask]
    ).mean()

    p_val = psnr(pred_norm.unsqueeze(0), clean_norm.unsqueeze(0), data_range=1.0)
    s_val = ssim(pred_norm.unsqueeze(0), clean_norm.unsqueeze(0), data_range=1.0)

    return {
        "flux": flux_err.item(),
        "mae_src": source_mae.item(),
        "psnr": p_val.item(),
        "ssim": s_val.item(),
    }


def accumulate(acc, m):
    for k in acc:
        acc[k].append(m[k])

def save_comparison_plot(sample_idx, dirty, clean, predictions, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 2D projections
    # ============================================================

    d = dirty.cpu().mean(dim=0).numpy()
    c = clean.cpu().mean(dim=0).numpy()

    c_max = c.max() if c.max() > 0 else 1.0

    # ============================================================
    # Predictions -> numpy
    # ============================================================

    all_preds = {
        name: predictions[name].cpu().mean(dim=0).numpy()
        for name in predictions
    }

    # ============================================================
    # Separate FNO / LNO / CLEAN
    # ============================================================

    fno_methods = [
        name for name in all_preds
        if name.startswith("FNO") or name.startswith("PI-FNO")
    ]

    lno_methods = [
        name for name in all_preds
        if name.startswith("LNO") or name.startswith("PI-LNO")
    ]

    clean_method = "CLEAN" if "CLEAN" in all_preds else None

    # Mantieni l'ordine originale
    fno_methods = fno_methods[:6]
    lno_methods = lno_methods[:6]

    # ============================================================
    # Global residual range
    # ============================================================

    all_res_values = np.concatenate([
        np.abs(pred - c).ravel()
        for pred in all_preds.values()
    ])

    global_res_lim = np.percentile(
        all_res_values,
        99.5
    )

    global_res_lim = max(
        global_res_lim,
        1e-9
    )

    # ============================================================
    # Layout
    #
    # 4 rows:
    #
    #   row 0 = FNO predictions
    #   row 1 = FNO residuals
    #   row 2 = LNO predictions
    #   row 3 = LNO residuals
    #
    # 8 columns:
    #
    #   col 0   = Dirty / GT / CLEAN / CLEAN residual
    #   col 1-6 = six FNO/LNO methods
    #   col 7   = colorbar
    #
    # Schema:
    #
    #   Dirty       FNO1  FNO2  FNO3  FNO4  FNO5  FNO6
    #   GroundTruth R1    R2    R3    R4    R5    R6
    #   CLEAN       LNO1  LNO2  LNO3  LNO4  LNO5  LNO6
    #   CLEAN Res.  R1    R2    R3    R4    R5    R6
    # ============================================================

    cell_w = 1.5
    cell_h = 2.0

    fig_w = cell_w * 5
    fig_h = cell_h * 7

    fig = plt.figure(
        figsize=(fig_w, fig_h)
    )

    gs = gridspec.GridSpec(
        7,
        5,
        figure=fig,

        width_ratios=[
            1, 1, 1, 1, 0.07
        ],

        hspace=0.20,
        wspace=0.025,

        left=0.045,
        right=0.965,
        top=0.88,
        bottom=0.05,
    )

    # ============================================================
    # Column 0
    #
    # Dirty
    # Ground Truth
    # CLEAN
    # CLEAN residual
    # ============================================================

    # ------------------------------------------------------------
    # Dirty
    # ------------------------------------------------------------

    ax_dirty = fig.add_subplot(
        gs[0, 0]
    )

    ax_dirty.imshow(
        d,
        origin="lower",
        cmap="inferno",
        vmin=d.min(),
        vmax=d.max()
    )

    ax_dirty.set_title(
        "Dirty (Input)",
        fontsize=8,
        fontweight="bold"
    )

    ax_dirty.axis("off")

    # ------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------

    ax_gt = fig.add_subplot(
        gs[0, 1]
    )

    ax_gt.imshow(
        c,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=c_max
    )

    ax_gt.set_title(
        "Ground Truth",
        fontsize=8,
        fontweight="bold"
    )

    ax_gt.axis("off")

    # ------------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------------

    if clean_method is not None:

        clean_pred = all_preds[clean_method]
        clean_res = clean_pred - c

        ax_clean = fig.add_subplot(
            gs[0, 2]
        )

        ax_clean.imshow(
            clean_pred,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=c_max
        )

        ax_clean.set_title(
            "CLEAN - Pred",
            fontsize=8,
            fontweight="bold"
        )

        ax_clean.axis("off")

        # --------------------------------------------------------
        # CLEAN residual
        # --------------------------------------------------------

        ax_clean_res = fig.add_subplot(
            gs[0, 3]
        )

        ax_clean_res.imshow(
            clean_res,
            origin="lower",
            cmap="RdBu_r",
            vmin=-global_res_lim,
            vmax=global_res_lim
        )

        ax_clean_res.set_title(
                    "CLEAN - Res",
                    fontsize=8,
                    fontweight="bold"
                )

        ax_clean_res.axis("off")

    else:

        # Se CLEAN non esiste, lascia vuote le due celle
        ax = fig.add_subplot(gs[0, 2])
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 3])
        ax.axis("off")

    # ============================================================
    # Row labels
    # ============================================================

    # fig.text(
    #     0.008,
    #     0.76,
    #     "Prediction",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.57,
    #     "Residual",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.34,
    #     "Prediction",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.15,
    #     "Residual",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # ============================================================
    # Plot helper
    # ============================================================

    im_pred_last = None
    im_res_last = None

    def plot_group(methods, pred_col, res_col):

        nonlocal im_pred_last
        nonlocal im_res_last

        for row, name in enumerate(methods, start=1):

            pred_np = all_preds[name]
            res_np = pred_np - c

            # ----------------------------------------------------
            # Prediction
            # ----------------------------------------------------

            ax_p = fig.add_subplot(
                gs[row, pred_col]
            )

            im_pred = ax_p.imshow(
                pred_np,
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=c_max
            )

            short = name.replace(
                "+TTO",
                " + TTO"
            )

            ax_p.set_title(
                short + " - Pred",
                fontsize=7.5,
                fontweight="bold"
            )

            ax_p.axis("off")

            im_pred_last = im_pred

            # ----------------------------------------------------
            # Residual
            # ----------------------------------------------------

            ax_r = fig.add_subplot(
                gs[row, res_col]
            )

            im_res = ax_r.imshow(
                res_np,
                origin="lower",
                cmap="RdBu_r",
                vmin=-global_res_lim,
                vmax=global_res_lim
            )

            ax_r.axis("off")

            ax_r.set_title(
                            short + " - Res",
                            fontsize=7.5,
                            fontweight="bold"
                        )

            im_res_last = im_res

    # ============================================================
    # FNO methods
    # ============================================================

    plot_group(
        fno_methods,
        pred_col=0,
        res_col=1
    )

    # ============================================================
    # LNO methods
    # ============================================================

    plot_group(
        lno_methods,
        pred_col=2,
        res_col=3
    )

    # ============================================================
    # Colorbars
    # ============================================================

    # ------------------------------------------------------------
    # Prediction colorbar
    # ------------------------------------------------------------

    if im_pred_last is not None:

        pos_pred = fig.add_axes([
            0.972,
            0.51,
            0.012,
            0.34
        ])

        cbar_pred = fig.colorbar(
            im_pred_last,
            cax=pos_pred
        )

        cbar_pred.set_label(
            "Flux [Jy/px²]",
            fontsize=8
        )

        cbar_pred.ax.tick_params(
            labelsize=7
        )

    # ------------------------------------------------------------
    # Residual colorbar
    # ------------------------------------------------------------

    if im_res_last is not None:

        pos_res = fig.add_axes([
            0.972,
            0.10,
            0.012,
            0.34
        ])

        cbar_res = fig.colorbar(
            im_res_last,
            cax=pos_res
        )

        cbar_res.set_label(
            "Pred − GT",
            fontsize=8
        )

        cbar_res.ax.tick_params(
            labelsize=7
        )

    # ============================================================
    # Separator between input/CLEAN and neural methods
    # ============================================================

    fig.text(
        0.145,
        0.50,
        "",
        va="center"
    )

    # # ============================================================
    # # Section labels
    # # ============================================================

    # fig.text(
    #     0.49,
    #     0.94,
    #     ha="center",
    #     va="center",
    #     fontsize=11,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.49,
    #     0.48,
    #     ha="center",
    #     va="center",
    #     fontsize=11,
    #     fontweight="bold"
    # )

    # ============================================================
    # Title
    # ============================================================

    plt.suptitle(
        f"Methods Comparison — Sample {sample_idx}",
        fontsize=13,
        fontweight="bold",
        y=0.975
    )

    # ============================================================
    # Save
    # ============================================================

    fname = (
        f"comparison_sample_{sample_idx:03d}.png"
    )

    path = os.path.join(
        output_dir,
        fname
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  → Saved: {path}"
    )

def save_comparison_plot_1(sample_idx, dirty, clean, predictions, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 2D projections
    # ============================================================

    d = dirty.cpu().mean(dim=0).numpy()
    c = clean.cpu().mean(dim=0).numpy()

    c_max = c.max() if c.max() > 0 else 1.0

    # ============================================================
    # Predictions -> numpy
    # ============================================================

    all_preds = {
        name: predictions[name].cpu().mean(dim=0).numpy()
        for name in predictions
    }

    # ============================================================
    # Separate FNO / LNO / CLEAN
    # ============================================================

    fno_methods = [
        name for name in all_preds
        if name.startswith("FNO") or name.startswith("PI-FNO")
    ]

    lno_methods = [
        name for name in all_preds
        if name.startswith("LNO") or name.startswith("PI-LNO")
    ]

    clean_method = "CLEAN" if "CLEAN" in all_preds else None

    # Mantieni l'ordine originale
    fno_methods = fno_methods[:6]
    lno_methods = lno_methods[:6]

    # ============================================================
    # Global residual range
    # ============================================================

    all_res_values = np.concatenate([
        np.abs(pred - c).ravel()
        for pred in all_preds.values()
    ])

    global_res_lim = np.percentile(
        all_res_values,
        99.5
    )

    global_res_lim = max(
        global_res_lim,
        1e-9
    )

    # ============================================================
    # Layout
    #
    # 4 rows:
    #
    #   row 0 = FNO predictions
    #   row 1 = FNO residuals
    #   row 2 = LNO predictions
    #   row 3 = LNO residuals
    #
    # 8 columns:
    #
    #   col 0   = Dirty / GT / CLEAN / CLEAN residual
    #   col 1-6 = six FNO/LNO methods
    #   col 7   = colorbar
    #
    # Schema:
    #
    #   Dirty       FNO1  FNO2  FNO3  FNO4  FNO5  FNO6
    #   GroundTruth R1    R2    R3    R4    R5    R6
    #   CLEAN       LNO1  LNO2  LNO3  LNO4  LNO5  LNO6
    #   CLEAN Res.  R1    R2    R3    R4    R5    R6
    # ============================================================

    cell_w = 1.5
    cell_h = 2.0

    fig_w = cell_w * 8
    fig_h = cell_h * 4

    fig = plt.figure(
        figsize=(fig_w, fig_h)
    )

    gs = gridspec.GridSpec(
        4,
        8,
        figure=fig,

        width_ratios=[
            1, 1, 1, 1, 1, 1, 1, 0.07
        ],

        hspace=0.20,
        wspace=0.025,

        left=0.045,
        right=0.965,
        top=0.88,
        bottom=0.05,
    )

    # ============================================================
    # Column 0
    #
    # Dirty
    # Ground Truth
    # CLEAN
    # CLEAN residual
    # ============================================================

    # ------------------------------------------------------------
    # Dirty
    # ------------------------------------------------------------

    ax_dirty = fig.add_subplot(
        gs[0, 0]
    )

    ax_dirty.imshow(
        d,
        origin="lower",
        cmap="inferno",
        vmin=d.min(),
        vmax=d.max()
    )

    ax_dirty.set_title(
        "Dirty (Input)",
        fontsize=8,
        fontweight="bold"
    )

    ax_dirty.axis("off")

    # ------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------

    ax_gt = fig.add_subplot(
        gs[1, 0]
    )

    ax_gt.imshow(
        c,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=c_max
    )

    ax_gt.set_title(
        "Ground Truth",
        fontsize=8,
        fontweight="bold"
    )

    ax_gt.axis("off")

    # ------------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------------

    if clean_method is not None:

        clean_pred = all_preds[clean_method]
        clean_res = clean_pred - c

        ax_clean = fig.add_subplot(
            gs[2, 0]
        )

        ax_clean.imshow(
            clean_pred,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=c_max
        )

        ax_clean.set_title(
            "CLEAN - Pred",
            fontsize=8,
            fontweight="bold"
        )

        ax_clean.axis("off")

        # --------------------------------------------------------
        # CLEAN residual
        # --------------------------------------------------------

        ax_clean_res = fig.add_subplot(
            gs[3, 0]
        )

        ax_clean_res.imshow(
            clean_res,
            origin="lower",
            cmap="RdBu_r",
            vmin=-global_res_lim,
            vmax=global_res_lim
        )

        ax_clean_res.set_title(
                    "CLEAN - Res",
                    fontsize=8,
                    fontweight="bold"
                )

        ax_clean_res.axis("off")

    else:

        # Se CLEAN non esiste, lascia vuote le due celle
        ax = fig.add_subplot(gs[2, 0])
        ax.axis("off")

        ax = fig.add_subplot(gs[3, 0])
        ax.axis("off")

    # ============================================================
    # Row labels
    # ============================================================

    # fig.text(
    #     0.008,
    #     0.76,
    #     "Prediction",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.57,
    #     "Residual",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.34,
    #     "Prediction",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.008,
    #     0.15,
    #     "Residual",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     fontweight="bold"
    # )

    # ============================================================
    # Plot helper
    # ============================================================

    im_pred_last = None
    im_res_last = None

    def plot_group(methods, pred_row, res_row):

        nonlocal im_pred_last
        nonlocal im_res_last

        for col, name in enumerate(methods, start=1):

            pred_np = all_preds[name]
            res_np = pred_np - c

            # ----------------------------------------------------
            # Prediction
            # ----------------------------------------------------

            ax_p = fig.add_subplot(
                gs[pred_row, col]
            )

            im_pred = ax_p.imshow(
                pred_np,
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=c_max
            )

            short = name.replace(
                "+TTO",
                " + TTO"
            )

            ax_p.set_title(
                short + " - Pred",
                fontsize=7.5,
                fontweight="bold"
            )

            ax_p.axis("off")

            im_pred_last = im_pred

            # ----------------------------------------------------
            # Residual
            # ----------------------------------------------------

            ax_r = fig.add_subplot(
                gs[res_row, col]
            )

            im_res = ax_r.imshow(
                res_np,
                origin="lower",
                cmap="RdBu_r",
                vmin=-global_res_lim,
                vmax=global_res_lim
            )

            ax_r.axis("off")

            ax_r.set_title(
                            short + " - Res",
                            fontsize=7.5,
                            fontweight="bold"
                        )

            im_res_last = im_res

    # ============================================================
    # FNO methods
    # ============================================================

    plot_group(
        fno_methods,
        pred_row=0,
        res_row=1
    )

    # ============================================================
    # LNO methods
    # ============================================================

    plot_group(
        lno_methods,
        pred_row=2,
        res_row=3
    )

    # ============================================================
    # Colorbars
    # ============================================================

    # ------------------------------------------------------------
    # Prediction colorbar
    # ------------------------------------------------------------

    if im_pred_last is not None:

        pos_pred = fig.add_axes([
            0.972,
            0.51,
            0.012,
            0.34
        ])

        cbar_pred = fig.colorbar(
            im_pred_last,
            cax=pos_pred
        )

        cbar_pred.set_label(
            "Flux [Jy/px²]",
            fontsize=8
        )

        cbar_pred.ax.tick_params(
            labelsize=7
        )

    # ------------------------------------------------------------
    # Residual colorbar
    # ------------------------------------------------------------

    if im_res_last is not None:

        pos_res = fig.add_axes([
            0.972,
            0.10,
            0.012,
            0.34
        ])

        cbar_res = fig.colorbar(
            im_res_last,
            cax=pos_res
        )

        cbar_res.set_label(
            "Pred − GT",
            fontsize=8
        )

        cbar_res.ax.tick_params(
            labelsize=7
        )

    # ============================================================
    # Separator between input/CLEAN and neural methods
    # ============================================================

    fig.text(
        0.145,
        0.50,
        "",
        va="center"
    )

    # # ============================================================
    # # Section labels
    # # ============================================================

    # fig.text(
    #     0.49,
    #     0.94,
    #     ha="center",
    #     va="center",
    #     fontsize=11,
    #     fontweight="bold"
    # )

    # fig.text(
    #     0.49,
    #     0.48,
    #     ha="center",
    #     va="center",
    #     fontsize=11,
    #     fontweight="bold"
    # )

    # ============================================================
    # Title
    # ============================================================

    plt.suptitle(
        f"Methods Comparison — Sample {sample_idx}",
        fontsize=13,
        fontweight="bold",
        y=0.975
    )

    # ============================================================
    # Save
    # ============================================================

    fname = (
        f"comparison_sample_{sample_idx:03d}.png"
    )

    path = os.path.join(
        output_dir,
        fname
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  → Saved: {path}"
    )

def save_comparison_plot_2(sample_idx, dirty, clean, predictions, output_dir):
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def save_metrics_chart(all_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    methods = list(all_metrics.keys())

    display_names = {
        "FNO2d": "FNO 2D",
        "FNO3d": "FNO 3D",
        "PI-FNO2d": "PI-FNO 2D",
        "PI-FNO2d+TTO": "PI-FNO 2D\n+ TTO",
        "PI-FNO3d": "PI-FNO 3D",
        "PI-FNO3d+TTO": "PI-FNO 3D\n+ TTO",
        "LNO2d": "LNO 2D",
        "LNO3d": "LNO 3D",
        "PI-LNO2d": "PI-LNO 2D",
        "PI-LNO2d+TTO": "PI-LNO 2D\n+ TTO",
        "PI-LNO3d": "PI-LNO 3D",
        "PI-LNO3d+TTO": "PI-LNO 3D\n+ TTO",
        "CLEAN": "CLEAN",
    }

    labels = [display_names.get(m, m) for m in methods]

    metric_info = [
        ("flux",    "Flux Error (%) ↓",      "#E07B54"),
        ("psnr",    "PSNR (dB) ↑",            "#4E9AB3"),
        ("ssim",    "SSIM ↑",                 "#6ABF7B"),
        ("time_ms", "Time per Sample (ms) ↓", "#B07CC6"),
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 11)
    )

    axes = axes.ravel()

    x = np.arange(len(methods))
    width = 0.65

    for ax, (key, title, color) in zip(axes, metric_info):

        means = np.array([
            all_metrics[m][key]
            for m in methods
        ])

        stds = np.array([
            all_metrics[m][f"{key}_std"]
            for m in methods
        ])

        bars = ax.bar(
            x,
            means,
            width=width,
            color=color,
            alpha=0.85,
            yerr=stds,
            capsize=4,
            error_kw={
                "elinewidth": 1.2,
                "ecolor": "#444444"
            }
        )

        # -----------------------------------------
        # TTO: tratteggio
        # -----------------------------------------
        for bar, method in zip(bars, methods):
            if "+TTO" in method:
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.8)

        # -----------------------------------------
        # Valori sopra le barre
        # -----------------------------------------
        ymax = np.nanmax(means + stds)

        for bar, mean, std in zip(bars, means, stds):

            offset = max(
                ymax * 0.015,
                1e-6
            )

            if key == "time_ms":
                value_text = f"{mean:.0f}"
            elif key == "ssim":
                value_text = f"{mean:.3f}"
            else:
                value_text = f"{mean:.2f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + offset,
                value_text,
                ha="center",
                va="bottom",
                fontsize=8.5
            )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
            fontsize=9
        )

        ax.grid(
            axis="y",
            alpha=0.25,
            linestyle="--"
        )

        ax.set_axisbelow(True)

        # -----------------------------------------
        # TIME: scala logaritmica
        # -----------------------------------------
        if key == "time_ms":

            ax.set_yscale("log")

            positive = means[means > 0]

            ymin = np.nanmin(positive) * 0.7
            ymax_log = np.nanmax(means + stds) * 2.0

            ax.set_ylim(
                bottom=max(ymin, 1e-1),
                top=ymax_log
            )

            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(
                    base=10,
                    numticks=8
                )
            )

            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda y, _: f"{y:g}"
                )
            )

        else:
            ax.set_ylim(bottom=0)

    # -----------------------------------------
    # Legenda TTO
    # -----------------------------------------
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="white",
            edgecolor="#333333",
            label="Model"
        ),
        Patch(
            facecolor="white",
            edgecolor="#333333",
            hatch="//",
            label="Model + TTO"
        )
    ]

    axes[0].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        frameon=True
    )

    fig.suptitle(
        "Benchmark Metrics",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    path = os.path.join(
        output_dir,
        "metrics_summary.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Metrics chart salvato in: {path}")

def save_metrics_chart_2(all_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    methods = list(all_metrics.keys())
    n       = len(methods)
    x       = np.arange(n)
    width   = 0.55

    metric_keys   = ["flux", "psnr", "ssim", "time_ms"]
    metric_labels = ["Flux Error (%) ↓", "PSNR (dB) ↑", "SSIM ↑                         ", "Time per Sample (ms) ↓"]
    colors        = ["#e07b54", "#4e9ab3", "#6abf7b", "#b07cc6"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax, key, label, color in zip(axes, metric_keys, metric_labels, colors):
        means  = [all_metrics[m][key]           for m in methods]
        stds   = [all_metrics[m][f"{key}_std"]  for m in methods]

        bars = ax.bar(x, means, width, color=color, alpha=0.82,
                      yerr=stds, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#333333"})

        for xi, (mean, std) in enumerate(zip(means, stds)):
            ax.text(xi, mean + std + (max(means) * 0.02),
                    f"{mean:.3f}\n±{std:.3f}",
                    ha="center", va="bottom", fontsize=10, color="#222222")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, color='red')
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle("Metrics Summary", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Metrics chart salvato in: {path}")


def print_metrics_table(all_metrics):
    col = 26  # larghezza colonna "mean ± std"
    header = (f"\n{'Metodo':<20}"
              f"  {'Flux Error (%)':<{col}}"
              f"  {'MAE':<{col}}"
              f"  {'PSNR (dB)':<{col}}"
              f"  {'SSIM':<{col}}"
              f"  {'Time (ms)':<{col}}")
    print(header)
    print("─" * len(header))
    for name, m in all_metrics.items():
        def fmt(key):
            return f"{m[key]:.4f} ± {m[key+'_std']:.4f}"
        print(f"{name:<20}"
              f"  {fmt('flux'):<{col}}"
              f"  {fmt('mae_src'):<{col}}"
              f"  {fmt('psnr'):<{col}}"
              f"  {fmt('ssim'):<{col}}"
              f"  {fmt('time_ms'):<{col}}")
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
        "LNO2d":    (args.lno2d, load_lno2d, False, False, None),
        "LNO3d":    (args.lno3d, load_lno3d, True, False, None),
        "PI-LNO2d": (args.pilno2d, load_pilno2d, False, True,
                     (args.tto_epochs_pilno2d, args.tto_lr_pilno2d)),
        "PI-LNO3d": (args.pilno3d, load_pilno3d, True, True,
                     (args.tto_epochs_pilno3d, args.tto_lr_pilno3d)),
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

    acc = {m: {"flux": [], "mae_src": [], "psnr": [], "ssim": [], "time_ms": []} for m in method_names}
    n_valid = {m: 0 for m in method_names}
    timer = Timer(device)

    for sample_idx, (dirty, clean, uv_mask) in enumerate(test_loader):
        # if sample_idx == 5:
        #     break
        # dirty, clean, psf: [1, C, H, W]
        dirty_s = dirty[0]   # [C, H, W] 
        clean_s = clean[0]

        uv_mask = uv_mask.to(device)
        psf = uv_to_psf(uv_mask)

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
        pred_clean = hogbom_clean_batch(dirty.to(device), psf.to(device), n_iter=args.n_iter_clean)[0]
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
        vals = acc[m]
        n = len(vals["flux"])
        if n > 0:
            avg_metrics[m] = {}
            for k, v_list in vals.items():
                arr = np.array(v_list)
                avg_metrics[m][k]           = arr.mean()
                avg_metrics[m][f"{k}_std"]  = arr.std()

    print_metrics_table(avg_metrics)
    save_metrics_chart(avg_metrics, args.output_dir)
    print(f"\nVisualizations saved in: {os.path.join(args.output_dir, 'comparisons')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FNO2d/3d/PI (pre+post TTO) vs CLEAN")

    parser.add_argument("--fno2d",   type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/fno2d.pth", help="Path checkpoint FNO2d")
    parser.add_argument("--fno3d",   type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/fno3d.pth", help="Path checkpoint FNO3d")
    parser.add_argument("--pifno2d", type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/pifno2d.pth", help="Path checkpoint PI-FNO2d")
    parser.add_argument("--pifno3d", type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/pifno3d.pth", help="Path checkpoint PI-FNO3d")
    parser.add_argument("--lno2d",   type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/lno2d.pth", help="Path checkpoint LNO2d")
    parser.add_argument("--lno3d",   type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/lno3d.pth", help="Path checkpoint LNO3d")
    parser.add_argument("--pilno2d", type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/pilno2d.pth", help="Path checkpoint PI-LNO2d")
    parser.add_argument("--pilno3d", type=str, default="/data1/rtessitore/ALMA-PINO/checkpoints/pilno3d.pth", help="Path checkpoint PI-LNO3d")
    parser.add_argument("--n_iter_clean", type=int, default=500, help="Number of iterations for CLEAN")

    # FNO2D
    parser.add_argument("--modes_fno2d", type=int,   default=24)
    parser.add_argument("--width_fno2d", type=int,   default=256)

    # FNO3D
    parser.add_argument("--modes_fno3d", type=int,   default=16)
    parser.add_argument("--modes_z_fno3d", type=int, default=12)
    parser.add_argument("--width_fno3d", type=int,   default=64)
  

    # PI-FNO2D
    parser.add_argument("--modes_pifno2d", type=int,   default=32)
    parser.add_argument("--width_pifno2d", type=int,   default=256)

    # PI-FNO3D
    parser.add_argument("--modes_pifno3d", type=int,   default=8)
    parser.add_argument("--modes_z_pifno3d", type=int, default=12)
    parser.add_argument("--width_pifno3d", type=int,   default=32)

    # LNO2D
    parser.add_argument("--modes_lno2d", type=int,   default=12)
    parser.add_argument("--width_lno2d", type=int,   default=256)

    # LNO3D
    parser.add_argument("--modes_lno3d", type=int,   default=16)
    parser.add_argument("--modes_z_lno3d", type=int, default=12)
    parser.add_argument("--width_lno3d", type=int,   default=32)

    # PI-LNO2D
    parser.add_argument("--modes_pilno2d", type=int,   default=8)
    parser.add_argument("--width_pilno2d", type=int,   default=256)

    # PI-LNO3D
    parser.add_argument("--modes_pilno3d", type=int,   default=16)
    parser.add_argument("--modes_z_pilno3d", type=int, default=12)
    parser.add_argument("--width_pilno3d", type=int,   default=32)


    parser.add_argument("--fourier_layers", type=int,   default=4)
    parser.add_argument("--channels",       type=int,   default=64)
    parser.add_argument("--pad_ratio",      type=float, default=0.1)
    parser.add_argument("--act",            type=str,   default="gelu",
                        choices=["gelu", "relu", "tanh", "leaky_relu"])

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="/data1/rtessitore/almasim/alma_dataset/dataset",
                        help="Path to the ALMA dataset")

    # TTO
    parser.add_argument("--tto_epochs_pifno2d", type=int,   default=10,
                        help="TTO epochs for PI-FNO2d")
    parser.add_argument("--tto_lr_pifno2d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-FNO2d (Keep << lr training)")
    parser.add_argument("--tto_epochs_pifno3d", type=int,   default=10,
                        help="TTO epochs for PI-FNO3d")
    parser.add_argument("--tto_lr_pifno3d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-FNO3d (Keep << lr training)")
    parser.add_argument("--tto_epochs_pilno2d", type=int,   default=10,
                        help="TTO epochs for PI-LNO2d")
    parser.add_argument("--tto_lr_pilno2d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-LNO2d (Keep << lr training)")
    parser.add_argument("--tto_epochs_pilno3d", type=int,   default=10,
                        help="TTO epochs for PI-LNO3d")
    parser.add_argument("--tto_lr_pilno3d",     type=float, default=5e-6,
                        help="TTO learning rate for PI-LNO3d (Keep << lr training)")

    # Output
    parser.add_argument("--output_dir", type=str, default="final_bench")
    parser.add_argument("--n_viz",      type=int, default=5,
                        help="Number of samples for which to save comparative plots")

    args = parser.parse_args()
    run_benchmark(args)