"""Convert trusted legacy ``cfgs.pkl`` files to the canonical YAML format."""

import argparse
import pickle
from pathlib import Path

import torch

from genesis_tools.config import save_cfgs_yaml


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def convert_pickle_to_yaml(input_path, output_path):
    """Convert a trusted five-item Genesis configuration pickle to YAML."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("rb") as stream:
        # Pickle can execute code while loading. This function is intentionally
        # opt-in and must only be used with trusted local artifacts.
        configs = pickle.load(stream)

    if not isinstance(configs, (list, tuple)) or len(configs) != 5:
        raise ValueError(f"Expected five configurations in {input_path}")

    save_cfgs_yaml(output_path, *(_to_serializable(value) for value in configs))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a trusted legacy cfgs.pkl file to cfgs.yaml."
    )
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--outfile", type=str, default="cfgs.yaml")
    args = parser.parse_args()

    input_path = args.log_dir / "cfgs.pkl"
    output_path = args.log_dir / args.outfile
    print(f"Loading trusted pickle: {input_path}")
    convert_pickle_to_yaml(input_path, output_path)
    print(f"Saved YAML to: {output_path}")


if __name__ == "__main__":
    main()
