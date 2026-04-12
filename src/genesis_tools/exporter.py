#!/usr/bin/env python3

import os
from pathlib import Path
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import yaml


def dump_policy(env_class, log_dir, ckpt, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    print(f"Loading: {log_dir}")

    # load configs
    with open(Path(log_dir)/"cfgs.yaml", "r") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = yaml.safe_load(f).values()
    reward_cfg["reward_scales"] = {}

    # initialize Genesis
    gs.init()

    # create environment
    env = env_class(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        **env_kwargs,
    )

    # runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)

    obs, _ = env.reset()

    model = runner.alg.actor_critic.actor
    model.eval()

    # generate and set a dummy input
    example_obs = torch.randn(1, obs.shape[1], device=gs.device)
    traced_model = torch.jit.trace(model, example_obs)

    policy_path = os.path.join(log_dir, "policy_traced.pt")
    traced_model.save(policy_path)
    print(f"Saved traced policy to: {policy_path}")