import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os

def save_predictions(dirty, clean, pred, epoch, tot_loss, data_loss, phys_loss, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    c = dirty.shape[1] // 2 
    
    img_dirty = dirty[0, c].detach().cpu().numpy()
    img_clean = clean[0, c].detach().cpu().numpy()
    img_pred = pred[0, c].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    fig.suptitle(f'Epoca {epoch} | Loss Tot: {tot_loss:.5f} | Data (MSE): {data_loss:.5f} | Phys: {phys_loss:.5f}', 
                 fontsize=14, fontweight='bold')
    
    vmax_dirty = img_dirty.max()
    
    im0 = axes[0].imshow(img_dirty, cmap='magma', vmin=0, vmax=vmax_dirty)
    axes[0].set_title('Dirty Image (Input)')
    axes[0].axis('off')
    
    vmax_clean = img_clean.max()

    im1 = axes[1].imshow(img_pred, cmap='magma', vmin=0, vmax=vmax_clean)
    axes[1].set_title('PI-FNO Prediction')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(img_clean, cmap='magma', vmin=0, vmax=vmax_clean)
    axes[2].set_title('Ground Truth (Clean)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch:03d}_prediction.png'), dpi=150)
    plt.close() 