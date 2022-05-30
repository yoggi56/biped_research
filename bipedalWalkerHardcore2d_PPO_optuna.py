from typing import Any
from typing import Dict
from typing import Callable

import time
import gym
import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn
from utils import linear_schedule

N_TRIALS = 150
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e6)
EVAL_FREQ = 10000#int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 10
SEED = 42
# tensorboard_log="./vPPO_tensorboard/"
# device = "cpu"

ENV_ID = "BipedalWalkerHardcore-v3"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
    "seed": SEED,
    # "tensorboard_log": tensorboard_log,
    "verbose": 0, 
    # "device": device
}

# def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
#     """
#     Sampler for PPO hyperparams.
#     :param trial:
#     :return:
#     """
#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#     n_steps = 1024
#     gamma = 0.995
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     # Uncomment to enable learning rate schedule
#     ent_coef = 0.00138405
#     clip_range = 0.2
#     n_epochs = 5
#     gae_lambda = 0.95
#     max_grad_norm = 0.9
#     vf_coef = 0.62166
#     n_layers = trial.suggest_int("n_layers", 1, 3)
#     n_neurons = trial.suggest_categorical("n_neurons", [64, 128, 256])
#     # # Uncomment for gSDE (continuous actions)
#     log_std_init = -0.757015
#     # Uncomment for gSDE (continuous action)
#     sde_sample_freq = 64
#     # Orthogonal initialization
#     ortho_init = False
#     activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])

#     # TODO: account when using multiple envs
#     if batch_size > n_steps:
#         batch_size = n_steps

#     # Independent networks usually work best
#     # when not working with images
#     actor = [n_neurons]*n_layers
#     critic = [n_neurons]*n_layers
#     net_arch = [dict(pi=actor, vf=critic)]

#     activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

#     returned_value = {
#         "n_steps": n_steps,
#         "batch_size": batch_size,
#         "gamma": gamma,
#         "learning_rate": learning_rate,
#         "ent_coef": ent_coef,
#         "clip_range": clip_range,
#         "n_epochs": n_epochs,
#         "gae_lambda": gae_lambda,
#         "max_grad_norm": max_grad_norm,
#         "vf_coef": vf_coef,
#         "sde_sample_freq": sde_sample_freq,
#         "policy_kwargs": dict(
#             log_std_init=log_std_init,
#             net_arch=net_arch,
#             activation_fn=activation_fn,
#             ortho_init=ortho_init,
#         ),
#     }
#     print("Try to use these hyperparameters: {0}".format(returned_value))

#     return returned_value

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    #lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # #net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_neurons = trial.suggest_categorical("n_neurons", [100, 200, 300, 400, 500])
    # # Uncomment for gSDE (continuous actions)
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # # Uncomment for gSDE (continuous action)
    sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    #activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # # Independent networks usually work best
    # # when not working with images
    actor = [n_neurons]*n_layers
    critic = [n_neurons]*n_layers
    net_arch = [dict(pi=actor, vf=critic)]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    returned_value = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
    print("Try to use these hyperparameters: {0}".format(returned_value))

    return returned_value

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 20,
        eval_freq: int = 5000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.last_mean_reward > self.best_mean_reward:
                  self.best_mean_reward = self.last_mean_reward
            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {self.last_mean_reward:.2f}")

            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model
    model = PPO(**kwargs)
    # Create env used for evaluation
    eval_env = gym.make(ENV_ID)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print("GOVNO {0}".format(e))
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        print("nan encountered")
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.best_mean_reward

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize",
                        study_name="vPPO_bipedWalkerHardcore2d", storage="mysql://root:12345678@localhost/vppo_bipedwalkerHardcore2d", load_if_exists=True)
    
    start_time = time.time()
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass
    total_time = time.time() - start_time

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    print("Elapsed time: {0}".format(total_time))
