#!/usr/bin/env python3

import optuna
import argparse
from train import train_w_args
def objective(args, trial):
    args['learning_rate'] = trial.suggest_loguniform('learning_rate', .00001, .001)
    # y = trial.suggest_float('y', -1, 1)

    print(args)

    model = train_w_args(args)

    print(f"{model.eval()=}")

    raise ValueError

def param_optim(args):
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        load_if_exists=True,
        study_name="demo",
        sampler=optuna.samplers.TPESampler()
    )

    study.optimize(
        lambda t: objective(args, t),
        n_trials=20)

    print(f"{study.best_params=}")



if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model-type",
#         default="DQN",
#         choices=["DQN", "PPO", "SAC", "A2C"],
#         help="The model type to train.",
#     )
#     parser.add_argument(
#         "--total-timesteps",
#         default=100,
#         type=int,
#         help="The number of total timesteps to train for.",
#     )
#     parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
#     parser.add_argument(
#         "--realtime", action="store_true", help="If true, slows down to real time."
#     )
#     parser.add_argument(
#         "--log", action="store_true", help="If true, logs data to Tensorboard."
#     )
    args = {
        'model_type': 'DQN',
        'total_timesteps': 5,
        'seed': 42,
        'realtime': False,
        'log': False
    }
    param_optim(args)