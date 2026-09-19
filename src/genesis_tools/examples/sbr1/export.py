"""Export an SBR1 checkpoint as a TorchScript policy."""

import argparse
from copy import deepcopy
from pathlib import Path

import genesis as gs

from genesis_tools.config import load_cfgs_yaml
from genesis_tools.examples.sbr1.env import Sbr1Env
from genesis_tools.exporter import export_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    gs.init()
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs_yaml(
        args.log_dir / "cfgs.yaml"
    )
    reward_cfg = deepcopy(reward_cfg)
    reward_cfg["reward_scales"] = {}
    env = Sbr1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.viewer,
    )
    export_policy(env, train_cfg, args.log_dir, args.ckpt, gs.device)


if __name__ == "__main__":
    main()
