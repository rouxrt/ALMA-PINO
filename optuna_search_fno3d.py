import optuna
import sys
import os
import torch
import gc
from argparse import Namespace
from train_fno3d import main
from models.utils import Logger
from optuna.samplers import TPESampler

def objective(trial):

    # lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)

    modes_xy = trial.suggest_categorical("modes_xy", [8, 12, 16])
    modes_z = trial.suggest_categorical("modes_z", [4, 6, 8])
    # alpha = trial.suggest_float("alpha", 0.0, 0.5)

    # lambda_phys = trial.suggest_float("lambda_phys", 0.1, 20.0)

    # batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    # width = trial.suggest_categorical("width", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.01, 0.5)
    lambda_phys = trial.suggest_float("lambda_phys", 0.0, 5.0)
    width = trial.suggest_categorical("width", [16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    #fourier_layers = trial.suggest_categorical("fourier_layers", [2, 4, 6])

    print(f"\n{'='*60}")
    print(f"STARTING TRIAL {trial.number}")
    print(f"Parameters: LR={lr:.5f}, Modes_xy={modes_xy}, Modes_z={modes_z}, Alpha={alpha:.3f}, Phys={lambda_phys:.2f}, Width={width}, BS={batch_size}")
    print(f"{'='*60}\n")

    args = Namespace(
        dataset_path="dataset/simulations",
        num_samples=200,         
        channels=16,
        img_size=32,
        extended_source=True,
        modes_x=modes_xy,         
        modes_y=modes_xy,         
        modes_z=modes_z,                
        width=width,
        fourier_layers=4,
        pad_ratio=0.1,
        epochs=100,                
        batch_size=batch_size,    
        learning_rate=lr,         
        lambda_data=1.0,
        lambda_phys=lambda_phys,  
        lambda_spec=0.0,
        alpha=alpha,           
        act="gelu",   
        trial=trial               
    )

    try:
        best_val_l1_raw, best_val_flux = main(args)

        #trial.set_user_attr("val_loss", best_val_loss)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[!] GPU Out of Memory. Skipping trial.\n")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    finally: 
        if 'train_dataset' in locals(): del train_dataset
        if 'train_dataloader' in locals(): del train_dataloader
        torch.cuda.empty_cache()
        gc.collect()

    return best_val_l1_raw, best_val_flux

if __name__ == "__main__":
    plots_dir = os.path.join("optuna_results", "optuna_plots_3d")
    os.makedirs(plots_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(plots_dir, "log.txt"))

    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        sampler=sampler,
        directions= ["minimize","minimize"]
        # pruner=optuna.pruners.HyperbandPruner(
        #     min_resource=25,    
        #     max_resource=100,   
        #     reduction_factor=3 
        # )
    )

    print("Starting Bayesian Optimization with Optuna (TPE)...")
    
    study.optimize(objective, n_trials=100)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")



    # print(f"Best Flux Error Achieved (Validation): {study.best_value:.5f}")
    # best_val_loss = study.best_trial.user_attrs.get("val_loss", "N/A")
    # print(f"Validation Loss of the winning model: {best_val_loss:.5f}")
    # print("Hyperparameters of the winning configuration:")
    # for key, value in study.best_params.items():
    #     print(f"    --{key}: {value}")
    # print("="*50)
    
    pareto_front = study.best_trials
    print(f"\nFound {len(pareto_front)} Pareto-optimal trials:")
    print("="*50)
    

    print("\nBest Trials (Leaderboard):")
    
    # completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # completed_trials.sort(key=lambda t: t.value)
    
    # top_n = min(5, len(completed_trials))
    
    # for rank in range(top_n):
    #     t = completed_trials[rank]
    #     flux_err = t.value
    #     val_loss = t.user_attrs.get('val_loss', float('nan'))
        
    #     print(f"\n[Rank {rank + 1}] - Trial ID: {t.number}")
    #     print(f"    Val Loss   : {val_loss:.6f}")
    #     print(f"    Flux Error : {flux_err:.3f}%")
    #     print(f"    Parameters  : ", end="")
        
    #     params_str = ", ".join([f"{k}={v}" for k, v in t.params.items()])
    #     print(params_str)
    
    pareto_front.sort(key=lambda t: t.values[0])
    
    top_n = min(10, len(pareto_front))
    
    for rank in range(top_n):
        t = pareto_front[rank]
        l1_err = t.values[0]
        flux_err = t.values[1]
        
        print(f"\n[Pareto Solution {rank + 1}] - Trial ID: {t.number}")
        print(f"    L1 Error (Space): {l1_err:.6f}")
        print(f"    Flux Error (%)  : {flux_err:.3f}%")
        print(f"    Parameters      : ", end="")
        
        params_str = ", ".join([f"{k}={v}" for k, v in t.params.items()])
        print(params_str)
        sys.stdout.flush()

    print("\nSaving optuna plots...")
    plots_dir = os.path.join("optuna_results", "optuna_plots_3d")
    os.makedirs(plots_dir, exist_ok=True)

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_pareto_front
    )
    try:
        fig_pareto = plot_pareto_front(study, target_names=["L1 Error (Spatial)", "Flux Error (Global)"])
        fig_pareto.write_html(os.path.join(plots_dir, "pareto_front.html"))
        
        fig_importance_l1 = plot_param_importances(study, target=lambda t: t.values[0], target_name="L1 Error")
        fig_importance_l1.write_html(os.path.join(plots_dir, "parameter_importance_L1.html"))
        
        fig_importance_flux = plot_param_importances(study, target=lambda t: t.values[1], target_name="Flux Error")
        fig_importance_flux.write_html(os.path.join(plots_dir, "parameter_importance_Flux.html"))

        fig_slice_l1 = plot_slice(study, target=lambda t: t.values[0], target_name="L1 Error")
        fig_slice_l1.write_html(os.path.join(plots_dir, "slice_plot_l1.html"))

        fig_slice_flux = plot_slice(study, target=lambda t: t.values[1], target_name="Flux Error")
        fig_slice_flux.write_html(os.path.join(plots_dir, "slice_plot_flux.html"))
    except Exception as e:
        print(f"Impossible to generate some multi-objective plots: {e}")
        
    print(f"Plots saved to: {plots_dir}")

    # fig_history = plot_optimization_history(study)
    # fig_history.write_html(os.path.join(plots_dir, "optimization_history.html"))
    
    # fig_importance = plot_param_importances(study)
    # fig_importance.write_html(os.path.join(plots_dir, "parameter_importance.html"))
    
    # fig_slice = plot_slice(study)
    # fig_slice.write_html(os.path.join(plots_dir, "slice_plot.html"))
    
    # print(f"Plots saved to: {plots_dir}")
