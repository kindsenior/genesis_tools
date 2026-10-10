"""Small compatibility boundary for the repository's pinned rsl-rl API."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


RSL_RL_VERSION = "5.5.1"


def require_rsl_rl_version(expected=RSL_RL_VERSION):
    """Require the rsl-rl distribution used by the saved configuration format."""
    try:
        legacy_distribution = version("rsl-rl")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "The conflicting 'rsl-rl' distribution is installed "
            f"(version {legacy_distribution}); uninstall it and use "
            f"rsl-rl-lib=={expected}."
        )

    try:
        installed = version("rsl-rl-lib")
    except PackageNotFoundError as error:
        raise RuntimeError(f"rsl-rl-lib=={expected} is required but is not installed") from error

    if installed != expected:
        raise RuntimeError(f"rsl-rl-lib=={expected} is required; found {installed}")
    return installed


def require_actor_critic_checkpoint(path):
    """Reject legacy 2.2.4 checkpoints before rsl-rl emits a cryptic key error."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"actor_state_dict", "critic_state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"Legacy or unsupported rsl-rl checkpoint {path}: missing {', '.join(missing)}. "
            "Use the rsl-rl 2.2.4 environment for an old checkpoint, or train a new "
            f"policy with rsl-rl {RSL_RL_VERSION}."
        )
    return checkpoint


class TensorEnvAdapter:
    """Adapt a legacy tensor environment to rsl-rl 5 observation groups."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.env_cfg

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _as_tensordict(self, observations, extras):
        critic = (extras or {}).get("observations", {}).get("critic", observations)
        return TensorDict(
            {"policy": observations, "critic": critic},
            batch_size=[self.num_envs],
        )

    def get_observations(self):
        observations, extras = self.env.get_observations()
        return self._as_tensordict(observations, extras)

    def reset(self, envs_idx=None):
        observations, extras = self.env.reset(envs_idx)
        return self._as_tensordict(observations, extras), extras

    def step(self, actions):
        observations, rewards, dones, extras = self.env.step(actions)
        return self._as_tensordict(observations, extras), rewards, dones, extras


class TimedOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with checkpoint validation and an optional timing hook."""

    def __init__(self, *args, **kwargs):
        require_rsl_rl_version()
        super().__init__(*args, **kwargs)
        original_log = self.logger.log

        def log(*log_args, **log_kwargs):
            original_log(*log_args, **log_kwargs)
            timing = getattr(self, "training_timing", None)
            if timing is not None:
                timing.report(
                    log_kwargs["it"],
                    log_kwargs["collect_time"],
                    log_kwargs["learn_time"],
                )

        self.logger.log = log

    def load(self, path, *args, **kwargs):
        require_actor_critic_checkpoint(path)
        return super().load(str(path), *args, **kwargs)
