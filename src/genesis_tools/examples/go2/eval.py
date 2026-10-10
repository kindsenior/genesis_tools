#!/usr/bin/env -S python3 -i

import argparse
from pathlib import Path

import torch

import genesis as gs

from genesis_tools.config import load_cfgs_yaml
from genesis_tools.examples.go2.env import Go2Env
from genesis_tools.rsl import TimedOnPolicyRunner, require_rsl_rl_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="logs/go2-walking/test")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    # Require an explicit checkpoint to avoid evaluating the wrong model.
    parser.add_argument("--ckpt", type=int, required=True, help="checkpoint to load")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    require_rsl_rl_version()
    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)

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
        show_viewer=not args.headless,
        rendered_envs_idx=list(range(args.num_envs))
    )

    runner = TimedOnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = Path(log_dir)/f"model_{args.ckpt}.pt"
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    observations = env.reset()
    step = 0
    with torch.no_grad():
        while args.max_steps is None or step < args.max_steps:
            actions = policy(observations)
            observations, rews, dones, infos = env.step(actions)
            step += 1


if __name__ == "__main__":
    main()

"""
# evaluation
python scripts/go2_locomotion/go2_eval.py -l logs/go2_locomotion/test --ckpt 100
"""
