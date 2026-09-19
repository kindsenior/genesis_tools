"""Read and write the training configuration artifact shared with LibTorch."""

from collections import namedtuple
from pathlib import Path

import yaml


CONFIG_KEYS = ("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "train_cfg")
TrainingConfigs = namedtuple("TrainingConfigs", CONFIG_KEYS)


def load_cfgs_yaml(path):
    """Load and validate a ``cfgs.yaml`` file without relying on key order."""
    path = Path(path)
    with path.open("r") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    missing = [key for key in CONFIG_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing configuration keys in {path}: {', '.join(missing)}")

    return TrainingConfigs(*(data[key] for key in CONFIG_KEYS))


def save_cfgs_yaml(path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """Write the canonical YAML configuration artifact."""
    path = Path(path)
    data = dict(zip(CONFIG_KEYS, (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)))
    with path.open("w") as stream:
        yaml.safe_dump(data, stream, default_flow_style=False, sort_keys=False)
