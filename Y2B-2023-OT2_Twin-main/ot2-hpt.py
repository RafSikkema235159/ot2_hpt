import gym
from stable_baselines3 import PPO
import os
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from ot2_wrapper_final import OT2Env
from clearml import Task

task = Task.init(project_name="Mentor Group E/Group DMRM", task_name="ppo-hpt")

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

# Define sweep config
sweep_config = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "rollout/ep_len_mean"},
    "parameters": {
        "learning_rate": {"distribution": "uniform", "min": 0.3, "max": 0.8},
        # "n_steps": {"distribution": "int_uniform", "min": 128, "max": 512},
        # "batch_size": {"distribution": "int_uniform", "min": 32, "max": 256},
        # "gamma": {"distribution": "uniform", "min": 0.9, "max": 0.999},
    },
}

sweep_id = wandb.sweep(sweep_config, project="sweep_for_weights")

def main(config=None):
    run = wandb.init(config, sync_tensorboard=True)

    config = run.config

    learning_rate = config.learning_rate 
    # n_steps = config.n_steps 
    # batch_size = config.batch_size
    # gamma = config.gamma 

    env = OT2Env()
    env.reset()

    model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=1, tensorboard_log="./logs_final_hpt")

    model.learn(total_timesteps=2_000_000, reset_num_timesteps=False)

    run.finish()

wandb.agent(sweep_id, main)