from typing import Any
from typing import Dict
from typing import Callable

import time
import gym
import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
from utils import linear_schedule
import os
from stable_baselines3.common.monitor import Monitor
import csv
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

import base64
from pathlib import Path

from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from datetime import datetime


N_TRIALS = 50
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(3e6)
EVAL_FREQ = 10000#int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 10
SEED = 42
# tensorboard_log="./vPPO_tensorboard/"
# device = "cpu"

ENV_ID = "BipedalWalkerHardcore-v3"
noise_std = 0.1
log_dir = "./models/vDDPG-bipedalWalkerHardcore2d-model"
best_model_dir = "./models/vDDPG-bipedalWalkerHardcore2d-model/best"
study_name = "vDDPG_bipedWalkerHardcore2d"
storage = "mysql://root:12345678@localhost/vddpg_bipedwalkerHardcore2d"
video_prefix = "bipedWalkerHardcore_vDDPG"

reward_threshold = 300


DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "seed": SEED,
    # "tensorboard_log": tensorboard_log,
    "verbose": 0, 
    # "device": device
}

def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:

    batch_size = trial.suggest_categorical("batch_size", [256, 400, 500])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99])
    learning_rate = trial.suggest_loguniform("learning_rate", 2e-5, 5e-5)
    n_layers = 2
    n_neurons = trial.suggest_categorical("n_neurons", [400, 500, 600])
    activation_fn = "relu"
    buffer_size = trial.suggest_categorical("buffer_size", [200000, 600000, 1000000])
    learning_starts = 10000
    gradient_steps = -1
    train_freq = (1, "episode")

    net_arch = [n_neurons]*n_layers
   
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    returned_value = {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "gradient_steps": gradient_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "train_freq": train_freq,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }
    print("Try to use these hyperparameters: {0}".format(returned_value))

    return returned_value

# def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:

#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#     gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     n_layers = trial.suggest_int("n_layers", 1, 3)
#     n_neurons = trial.suggest_categorical("n_neurons", [64, 100, 200, 300, 400])
#     activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
#     buffer_size = 200000
#     learning_starts = 10000
#     gradient_steps = -1
#     train_freq = (1, "episode")

#     net_arch = [n_neurons]*n_layers
   
#     activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

#     returned_value = {
#         "batch_size": batch_size,
#         "buffer_size": buffer_size,
#         "learning_starts": learning_starts,
#         "gradient_steps": gradient_steps,
#         "gamma": gamma,
#         "learning_rate": learning_rate,
#         "train_freq": train_freq,
#         "policy_kwargs": dict(
#             net_arch=net_arch,
#             activation_fn=activation_fn,
#         ),
#     }
#     print("Try to use these hyperparameters: {0}".format(returned_value))

#     return returned_value


def objective(trial: optuna.Trial) -> float:
    # current datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    dir = "{0}/{1}".format(log_dir, dt_string)
    os.makedirs(dir, exist_ok=True)

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_ddpg_params(trial))
    # Create env used for evaluation
    eval_env = make_vec_env(ENV_ID, n_envs=1)
    eval_env = VecMonitor(eval_env, dir)
    env = make_vec_env(ENV_ID, n_envs=1)
    env = VecMonitor(env, dir)
    # env = gym.make(ENV_ID)
    # env = Monitor(env, dir)

    # The noise objects for DDPG
    n_actions = eval_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    # Create the RL model
    model = DDPG(env=env, action_noise=action_noise, **kwargs)

    def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
        """
        :param env_id: (str)
        :param model: (RL model)
        :param video_length: (int)
        :param prefix: (str)
        :param video_folder: (str)
        """
        eval_env = DummyVecEnv([lambda: gym.make(env_id)])
        # Start the video at step=0 and record 500 steps
        eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                    record_video_trigger=lambda step: step == 0, video_length=video_length,
                                    name_prefix=prefix)

        obs = eval_env.reset()
        for _ in range(video_length):
            action, _ = model.predict(obs)
            obs, _, _, _ = eval_env.step(action)

        # Close the video recorder
        eval_env.close()

    def evaluate(model, env):
        pass

    class TrialEvalCallback(EvalCallback):
        """Callback used for evaluating and reporting a trial."""

        def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 5000,
            deterministic: bool = True,
            verbose: int = 1,
            log_dir: str = "./models/vDDPG-bipedalWalkerHardcore2d-model",
        ):

            super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=deterministic,
                verbose=0,
            )
            self.trial = trial
            self.log_dir = log_dir
            self.save_path = os.path.join(self.log_dir, 'best_model')
            self.eval_idx = 0
            self.is_pruned = False
            self.best_mean_reward = -np.inf
            self.verbose = verbose
            self.s_reward = []
            self.s_timestep = []

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                super()._on_step()
                self.eval_idx += 1
                self.trial.report(self.last_mean_reward, self.eval_idx)

                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                episodes = len(y)
                self.s_reward.append(self.last_mean_reward)
                self.s_timestep.append(self.num_timesteps)

                if self.last_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = self.last_mean_reward

                if self.verbose > 0:
                    print(f"Episodes: {episodes}")
                    # print(f"Num timesteps: {self.num_timesteps}")
                    # print(f"Last mean reward per episode: {self.last_mean_reward:.2f}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}")
                    print(f"============================================")

                # Prune trial if need
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False

                # save model parameters if reward is greater than 300
                if self.last_mean_reward >= reward_threshold:
                    Yellow = "\033[0;33m"
                    NC = "\033[0m"
                    print("{0}Reward threshold achieved{1}".format(Yellow, NC))
                    print("Evaluating model....")
                    evals= evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True, render=False, callback=None,
                                    reward_threshold=None, return_episode_rewards=True)
                    mean_reward = np.mean(evals[0])
                    std_reward = 0
                    print(f"Evaluation over 100 Episodes: {mean_reward} ")
                    if mean_reward >= reward_threshold:
                        # create folder for best models
                        now = datetime.now()
                        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
                        best_dir = "{0}/{1}".format(best_model_dir, dt_string)
                        os.makedirs(best_dir, exist_ok=True)
                        self.model.save(best_dir)
                        # сохранить файл с гиперпараметрами, числом эпизодов, шагов, полученное вознаграждение 
                        print("Saving training info...")
                        filename = "{0}/training_info.txt".format(best_dir)
                        print(filename)
                        with open(filename, mode="w") as f:
                            f.write("Episodes: {0}\r\n".format(episodes))
                            f.write("Timesteps: {0}\r\n".format(self.num_timesteps))
                            f.write("Eval reward: {0}\r\n".format(mean_reward))
                            f.write("Info and Hyperparameters:\r\n")
                            for k, v in kwargs.items():
                                str = "    {0}: {1}\r\n".format(k, v)
                                f.write(str)
                        # сохранить кривую вознаграждения
                        print("Saving reward CSV-data...")
                        filename = "{0}/rewards.csv".format(best_dir)
                        with open(filename, mode='w') as reward_file:
                            reward_writer = csv.writer(reward_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                            for i in range(len(self.s_timestep)):
                                reward_writer.writerow([self.s_reward[i], self.s_timestep[i]])
                        # сохранить видео
                        print("Saving video...")
                        record_video(ENV_ID, model, video_length=1500, video_folder=best_dir, prefix=video_prefix)
                        print(f"MISSION COMPLETED")
                        print(f"Score: {mean_reward}+/-{std_reward} reached at Episode: {episodes} ")
                        return False
                    
            return True

    
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True, log_dir=dir
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
        #model.env.close()
        eval_env.close()
        env.close()

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
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=30, interval_steps=10)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize",
                        study_name=study_name, storage=storage, load_if_exists=True)
    
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

    