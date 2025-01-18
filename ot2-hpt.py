import gym
from stable_baselines3 import PPO
import os
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from ot2_wrapper_final import OT2Env
from clearml import Task
import typing_extensions
import tensorboard

os.environ['WANDB_API_KEY'] = 'af1a6039f20199fa6afe8c2022dde72b137ba944'

task = Task.init(project_name="Mentor Group E/Group DMRM", task_name="ppo-hpt_Denys")

# Define sweep config
sweep_config = {
    "method": "bayes",
    "name": "sweep_Denys",
    "metric": {"goal": "minimize", "name": "rollout/ep_len_mean"},
    "parameters": {
        "learning_rate": {"values": [3e-4, 1e-4, 5e-4, 1e-3, 8e-5]},
        # "n_steps": {"distribution": "int_uniform", "min": 128, "max": 512},
        # "batch_size": {"distribution": "int_uniform", "min": 32, "max": 256},
        # "gamma": {"distribution": "uniform", "min": 0.9, "max": 0.999},
    },
}

# Connect the dictionary to your CLEARML Task
parameters_dict = Task.current_task().connect(sweep_config)

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

sweep_id = wandb.sweep(parameters_dict, project="sweep_for_weights")

def main(config=None):
    run = wandb.init(config, sync_tensorboard=True)

    config = run.config

    learning_rate = config.learning_rate 
    # n_steps = config.n_steps 
    # batch_size = config.batch_size
    # gamma = config.gamma 

    env = OT2Env()
    env.reset(seed=42)

    model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=1, device="cuda", tensorboard_log="./logs_final_hpt")

    model.learn(total_timesteps=2_500_000, reset_num_timesteps=False)

    run.finish()

wandb.agent(sweep_id, main, count=5)