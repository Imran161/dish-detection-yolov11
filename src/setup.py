import argparse
from train import train_model
from hpo import run_hpo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpo', action='store_true')
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()

    if args.hpo:
        study = run_hpo(args.trials)
        best = study.best_trial
        print(f"Best mAP: {best.value}\nParams: {best.params}")
        print("Training with best params...")
        train_model(best.params)
    else:
        train_model({})
