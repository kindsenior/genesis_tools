#!/usr/bin/env -S python3 -i

import argparse
from pathlib import Path
import shutil

import genesis as gs

from genesis_tools.config import load_cfgs_yaml, save_cfgs_yaml
from genesis_tools.examples.go2.env import Go2Env
from genesis_tools.rsl import TimedOnPolicyRunner, require_rsl_rl_version


def get_train_cfg(exp_name="go2-walking"):
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
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
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
        # Genesis uses 10 degrees in its original Go2 sample. Keep 45 degrees here
        # to avoid premature termination when training on uneven terrain.
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
            "use_terrain": False,
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
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
        # "lin_vel_x_range": [0, 0.5],
        # "lin_vel_y_range": [-0.3, 0.3],
        # "ang_vel_range": [-0.5, 0.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=Path, default=Path("logs/go2-walking/test"))
    parser.add_argument("-B", "--num_envs", "--num-envs", type=int, default=4096)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--max_iterations", "--max-iterations", type=int, default=101)
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

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = TimedOnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    # load a pretrained model and optimizer for resume
    ## added to the original code for resume functionality
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
python scripts/go2_locomotion/go2_train.py -l logs/go2_locomotion/test
"""
