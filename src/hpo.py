import optuna
import joblib
import os
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from train import train_model


OUTPUT_OPT = 'outputs/optuna'

def run_hpo(trials: int):
    os.makedirs(OUTPUT_OPT, exist_ok=True)
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
            interval_steps=5
        )
    )
    study.optimize(objective, n_trials=trials)
    joblib.dump(study, os.path.join(OUTPUT_OPT, 'study.pkl'))

    plt.figure(figsize=(10,6))
    plot_optimization_history(study)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_OPT,'optimization_history.png'))
    plt.close()

    plt.figure(figsize=(10,6))
    plot_param_importances(study)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_OPT,'param_importances.png'))
    plt.close()

    return study

def objective(trial):
    params = {
        'epochs': 300,
        'lr0': trial.suggest_float('lr0',1e-5,1e-2,log=True),
        'weight_decay': trial.suggest_float('weight_decay',1e-6,1e-3),
        'dropout': trial.suggest_float('dropout',0.0,0.5),
    }
    result = train_model(params, trial)
    return result.results_dict['metrics/mAP50-95(B)']