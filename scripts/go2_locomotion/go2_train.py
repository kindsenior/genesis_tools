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

from go2_env import Go2Env


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
            "init_noise_std": 1.0,
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
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        # "termination_if_roll_greater_than": 10,  # degree
        # "termination_if_pitch_greater_than": 10,
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # terrain
        "terrain": {
            "n_subterrains": [5, 5],
            "subterrain_size": [12.0, 12.0],
            "horizontal_scale": 0.25,
            "vertical_scale": 0.005,
            # "vertical_scale": 0.001,
            # "vertical_scale": 0.0005,
            # "vertical_scale": 0.0001,
            "subterrain_types": [
                # # withoug stepping stones
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # ["flat_terrain","pyramid_stairs_terrain",   "random_uniform_terrain",     "pyramid_stairs_terrain", "flat_terrain"],
                # ["flat_terrain","pyramid_sloped_terrain", "flat_terrain",               "wave_terrain",           "flat_terrain"],
                # ["flat_terrain","sloped_terrain",         "discrete_obstacles_terrain", "stairs_terrain",         "flat_terrain"],
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # withoug slope and stair
                ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                ["flat_terrain","wave_terrain",           "random_uniform_terrain",     "discrete_obstacles_terrain", "flat_terrain"],
                ["flat_terrain","pyramid_sloped_terrain", "flat_terrain",               "wave_terrain",               "flat_terrain"],
                ["flat_terrain","random_uniform_terrain", "discrete_obstacles_terrain", "pyramid_sloped_terrain",     "flat_terrain"],
                ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # # all flat terrain
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
                # ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
            ],
            "randomize": True,
        },
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        # "lin_vel_x_range": [0.5, 0.5],
        # "lin_vel_y_range": [0, 0],
        # "ang_vel_range": [0, 0],
        "lin_vel_x_range": [0, 0.5],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-0.5, 0.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="logs/go2_locomotion/test")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("-r", "--resume", type=bool, default=False)
    parser.add_argument("-rp", "--resume_path", type=str, default=None)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"{args.log_dir}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    args.max_iterations = args.max_iterations + 1
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

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # load a pretrained model and optimizer for resume
    if args.resume:
        # use parent path of log_dir as resume_path
        resume_path = Path(log_dir).parent / f"model_{args.ckpt}.pt" if args.resume_path is None else args.resume_path
        print(f"resume from {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
