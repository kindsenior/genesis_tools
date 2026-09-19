#!/usr/bin/env -S python3 -i

import argparse
from pathlib import Path
from importlib import metadata

import torch

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

from genesis_tools.config import load_cfgs_yaml
from genesis_tools.examples.go2.env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="logs/go2-walking/test")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    # Require an explicit checkpoint to avoid evaluating the wrong model.
    parser.add_argument("--ckpt", type=int, required=True, help="checkpoint to load")
    args = parser.parse_args()

    gs.init()

    log_dir = f"{args.log_dir}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs_yaml(
        Path(log_dir) / "cfgs.yaml"
    )
    reward_cfg["reward_scales"] = {}

    global env
    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        rendered_envs_idx=list(range(args.num_envs))
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = Path(log_dir)/f"model_{args.ckpt}.pt"
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python scripts/go2_locomotion/go2_eval.py -l logs/go2_locomotion/test --ckpt 100
"""
