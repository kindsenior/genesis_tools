#!/usr/bin/env -S python3 -i

import argparse
import os
from pathlib import Path
import yaml
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from sbr1_env import Sbr1Env
import numpy as np

def get_train_cfg(max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            # "noise_std_type": "log",
            "init_noise_std": 1.0, # 1.0 for reduce vibrations
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 25,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "hip_pitch_left_joint": 0.3,
            "hip_roll_left_joint": 0.0,
            "hip_yaw_left_joint": 0.0,
            "knee_pitch_left_joint": -0.5,
            "ankle_pitch_left_joint": 0.2,
            "ankle_roll_left_joint": 0.0,
            "hip_pitch_right_joint": 0.3,
            "hip_roll_right_joint": 0.0,
            "hip_yaw_right_joint": 0.0,
            "knee_pitch_right_joint": -0.5,
            "ankle_pitch_right_joint": 0.2,
            "ankle_roll_right_joint": 0.0,
        },
        "joint_names": [
            "hip_pitch_left_joint",
            "hip_roll_left_joint",
            "hip_yaw_left_joint",
            "knee_pitch_left_joint",
            "ankle_pitch_left_joint",
            "ankle_roll_left_joint",
            "hip_pitch_right_joint",
            "hip_roll_right_joint",
            "hip_yaw_right_joint",
            "knee_pitch_right_joint",
            "ankle_pitch_right_joint",
            "ankle_roll_right_joint",
        ],
        # PD
        "kp": 100.0,
        "kd": 30.0,
        "pdgain_rate": [
            1.0, 1.0, 1.0, 1.0, 0.5, 0.1,
            1.0, 1.0, 1.0, 1.0, 0.5, 0.1,
        ],
        # termination
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.33], # original
        "base_init_quat": [float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "pos": [0.0, 1.0, 1.0],
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 1.33,
        "feet_height_target": 0.2,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "lin_vel_z": -1.0,
            "base_y": -1.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "base_orientation": 1.0,
            "feet_y": -3.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        # "lin_vel_x_range": [0.3, 0.3],
        "lin_vel_x_range": [2.5, 3.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [-0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="logs/sbr1_locomotion/test")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"{args.log_dir}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.yaml", "w") as f:
        yaml.dump(
            {
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "train_cfg": train_cfg,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    env = Sbr1Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    os.system('say -v Ava "learning finished"')

if __name__ == "__main__":
    main()

"""
# training
python sbr1_train.py
"""
