import optuna
import sys
import os
from argparse import Namespace
from train_fno3d import main
from models.utils import Logger

def objective(trial):

    lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)

    modes_xy = trial.suggest_categorical("modes_xy", [8, 12, 16])
    modes_z = trial.suggest_categorical("modes_z", [4, 6, 8])

    alpha = trial.suggest_float("alpha", 0.0, 0.5)

    lambda_phys = trial.suggest_float("lambda_phys", 0.1, 20.0)

    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    width = trial.suggest_categorical("width", [16, 32, 64])

    act = trial.suggest_categorical("act", ["gelu", "relu", "tanh", "leaky_relu"])

    print(f"\n{'='*60}")
    print(f"STARTING TRIAL {trial.number}")
    print(f"Parameters: LR={lr:.5f}, Modes_xy={modes_xy}, Modes_z={modes_z}, Alpha={alpha:.3f}, Phys={lambda_phys:.2f}, Width={width}, BS={batch_size}, Act={act}")
    print(f"{'='*60}\n")

    args = Namespace(
        num_samples=200,         
        channels=16,
        img_size=32,
        extended_source=True,
        modes_x=modes_xy,         
        modes_y=modes_xy,         
        modes_z=modes_z,                
        width=width,
        fourier_layers=4,
        pad_ratio=0.0,
        epochs=30,                
        batch_size=batch_size,    
        learning_rate=lr,         
        lambda_data=1.0,
        lambda_phys=lambda_phys,  
        lambda_spec=0.0,
        alpha=alpha,           
        act=act,   
        trial=trial               
    )

    try:
        best_val_loss, best_val_flux = main(args)

        trial.set_user_attr("flux_error", best_val_flux)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[!] GPU Out of Memory. Skipping trial.\n")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    return best_val_loss

if __name__ == "__main__":
    os.makedirs('optuna_results', exist_ok=True)
    plots_dir = os.path.join("optuna_results", "optuna_plots_3d")
    sys.stdout = Logger(f"optuna_results/optuna_plots_3d/log.txt")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,   
            n_warmup_steps=5,     
            interval_steps=1      
        )
    )

    print("Starting Bayesian Optimization with Optuna (TPE)...")
    
    study.optimize(objective, n_trials=100)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")


    print(f"Best Loss Achieved (Validation): {study.best_value:.5f}")
    best_flux = study.best_trial.user_attrs.get("flux_error", "N/A")
    print(f"Flux Error of the winning model: {best_flux:.3f}%")
    print("Hyperparameters of the winning configuration:")
    for key, value in study.best_params.items():
        print(f"    --{key}: {value}")
    print("="*50)

    

    print("\nBest 5 Trials (Leaderboard):")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    completed_trials.sort(key=lambda t: t.value)
    
    top_n = min(5, len(completed_trials))
    
    for rank in range(top_n):
        t = completed_trials[rank]
        val_loss = t.value
        flux_err = t.user_attrs.get('flux_error', float('nan'))
        
        print(f"\n[Rank {rank + 1}] - Trial ID: {t.number}")
        print(f"    Val Loss   : {val_loss:.6f}")
        print(f"    Flux Error : {flux_err:.3f}%")
        print(f"    Parameters  : ", end="")
        
        params_str = ", ".join([f"{k}={v}" for k, v in t.params.items()])
        print(params_str)


    print("\nSaving optuna plots...")
    plots_dir = os.path.join("optuna_results", "optuna_plots_3d")
    os.makedirs(plots_dir, exist_ok=True)

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice
    )
    
    fig_history = plot_optimization_history(study)
    fig_history.write_html(os.path.join(plots_dir, "optimization_history.html"))
    
    fig_importance = plot_param_importances(study)
    fig_importance.write_html(os.path.join(plots_dir, "parameter_importance.html"))
    
    fig_slice = plot_slice(study)
    fig_slice.write_html(os.path.join(plots_dir, "slice_plot.html"))
    
    print(f"Plots saved to: {plots_dir}")
