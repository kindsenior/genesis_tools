#!/usr/bin/env -S python3 -i

import argparse
from pathlib import Path
import shutil

import genesis as gs

from genesis_tools.config import load_cfgs_yaml, save_cfgs_yaml
from genesis_tools.examples.sbr1.env import Sbr1Env
from genesis_tools.rsl import TensorEnvAdapter, TimedOnPolicyRunner, require_rsl_rl_version
import numpy as np

def get_train_cfg(exp_name="sbr1-locomotion"):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
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
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
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
        "lin_vel_x_range": [0, 1.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [-0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=Path, default=Path("logs/sbr1_locomotion/test"))
    parser.add_argument("-B", "--num_envs", "--num-envs", type=int, default=4096)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--max_iterations", "--max-iterations", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing log directory before training.",
    )
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-rp", "--resume_path", type=Path, default=None)
    args = parser.parse_args()

    require_rsl_rl_version()
    log_dir = args.log_dir
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    if args.resume and args.ckpt is None and args.resume_path is None:
        parser.error("--resume requires --ckpt or --resume_path")
    if log_dir.exists() and not args.overwrite and not args.resume:
        parser.error(
            f"Log directory already exists: {log_dir}\n"
            "Specify a different -l/--log_dir, or use --overwrite "
            "to delete the existing results."
        )

    if args.resume:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs_yaml(
            log_dir / "cfgs.yaml"
        )
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(log_dir.name)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    if args.overwrite and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        save_cfgs_yaml(
            log_dir / "cfgs.yaml",
            env_cfg,
            obs_cfg,
            reward_cfg,
            command_cfg,
            train_cfg,
        )

    env = Sbr1Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = TimedOnPolicyRunner(TensorEnvAdapter(env), train_cfg, str(log_dir), device=gs.device)

    if args.resume:
        resume_path = log_dir / f"model_{args.ckpt}.pt" if args.resume_path is None else args.resume_path
        if not resume_path.is_file():
            parser.error(f"Checkpoint not found: {resume_path}")
        print(f"resume from {resume_path}")
        runner.load(resume_path, map_location=str(gs.device))

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# training
python sbr1_train.py
"""
