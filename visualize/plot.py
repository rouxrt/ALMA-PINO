import matplotlib.pyplot as plt
import os
import sys


def save_predictions(dirty, clean, pred, epoch, tot_loss, data_loss, phys_loss, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    c = dirty.shape[1] // 2 
    
    img_dirty = dirty[0, c].detach().cpu().numpy()
    img_clean = clean[0, c].detach().cpu().numpy()
    img_pred = pred[0, c].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    fig.suptitle(f'Epoca {epoch} | Loss Tot: {tot_loss:.5f} | Data (MSE): {data_loss:.5f} | Phys: {phys_loss:.5f}', 
                 fontsize=14, fontweight='bold')
    
    global_vmin = min(dirty[0].min(), clean[0].min(), pred[0].min()).item()
    global_vmax = max(dirty[0].max(), clean[0].max(), pred[0].max()).item()
    
    im0 = axes[0].imshow(img_dirty, cmap='magma', vmin=global_vmin, vmax=global_vmax)
    axes[0].set_title('Dirty Image (Input)')
    axes[0].axis('off')
    
    

    im1 = axes[1].imshow(img_pred, cmap='magma', vmin=global_vmin, vmax=global_vmax)
    axes[1].set_title('PI-FNO Prediction')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(img_clean, cmap='magma', vmin=global_vmin, vmax=global_vmax)
    axes[2].set_title('Ground Truth (Clean)')
    axes[2].axis('off')
    
    fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)

    plt.savefig(os.path.join(output_dir, f'epoch_{epoch:03d}_prediction.png'), dpi=150)
    plt.close() 

def visualize_datacube(dirty, clean, pred, output_dir = "datacube"):
    os.makedirs(output_dir, exist_ok=True)

    global_vmin = min(dirty[0].min(), clean[0].min(), pred[0].min()).item()
    global_vmax = max(dirty[0].max(), clean[0].max(), pred[0].max()).item()

    for c in range(dirty.shape[1]):
        img_dirty = dirty[0, c].detach().cpu().numpy()
        img_clean = clean[0, c].detach().cpu().numpy()
        img_pred = pred[0, c].detach().cpu().numpy()


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(img_dirty, cmap='magma', origin='lower', vmin=global_vmin, vmax=global_vmax)
        axes[0].set_title(f'Dirty Image - Canale {c}')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(img_pred, cmap='magma', origin='lower', vmin=global_vmin, vmax=global_vmax)
        axes[1].set_title(f'PI-FNO Prediction - Canale {c}')
        axes[1].axis('off')
        
        im2 = axes[2].imshow(img_clean, cmap='magma', origin='lower', vmin=global_vmin, vmax=global_vmax)
        axes[2].set_title(f'Ground Truth (Clean) - Canale {c}')
        axes[2].axis('off')

        fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        
        plt.savefig(os.path.join(output_dir, f'channel_{c:02d}.png'), dpi=150)
        plt.close()


def plot_loss_history(history: dict, title: str, save_path: str):

    if not history:
        return
        
    epochs = range(len(history["train"]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history["train"], color='blue', label='Train')
    ax1.plot(epochs, history["val"], color='orange', label='Validation')
    ax1.set_title(f"MSE vs Epochs - {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    
    ax2.plot(epochs, history["data"], color='green', label='Data Loss (MSE)')
    ax2.plot(epochs, history["phys"], color='red', label='Physics Loss (MSE)')
    ax2.plot(epochs, history["train"], color='blue', label='Total Loss')    
    ax2.set_title(f"Loss Components vs Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Value")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

