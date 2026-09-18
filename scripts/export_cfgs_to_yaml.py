#!/usr/bin/env python3

import pickle
import yaml
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True, help='Path to logs/xxx directory (contains cfgs.pkl)')
    parser.add_argument('--outfile', type=str, default="cfgs.yaml", help='Output YAML filename')
    args = parser.parse_args()

    cfgs_path = os.path.join(args.log_dir, "cfgs.pkl")
    output_path = os.path.join(args.log_dir, args.outfile)

    print(f"Loading: {cfgs_path}")
    with open(cfgs_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    all_cfgs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
    }

    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    all_cfgs = to_serializable(all_cfgs)

    with open(output_path, "w") as f:
        yaml.safe_dump(all_cfgs, f, sort_keys=False)

    print(f"Saved YAML to: {output_path}")

if __name__ == "__main__":
    main()
