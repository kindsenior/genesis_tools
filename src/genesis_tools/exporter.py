#!/usr/bin/env python3

from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


def export_policy(env, train_cfg, log_dir, ckpt, device, output_path=None):
    """Export an rsl-rl actor as a TorchScript module for LibTorch."""
    log_dir = Path(log_dir)
    print(f"Loading: {log_dir}")

    # runner
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    resume_path = log_dir / f"model_{ckpt}.pt"
    runner.load(resume_path)

    obs, _ = env.reset()

    model = runner.alg.actor_critic.actor
    model.eval()

    # Trace the actor with one observation using the environment's dtype/device.
    with torch.inference_mode():
        traced_model = torch.jit.trace(model, obs[:1])

    if output_path is None:
        output_path = log_dir / "policy_traced.pt"
    output_path = Path(output_path)
    traced_model.save(str(output_path))
    print(f"Saved traced policy to: {output_path}")
    return output_path
