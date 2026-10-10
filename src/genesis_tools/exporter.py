#!/usr/bin/env python3

from pathlib import Path

from genesis_tools.rsl import TimedOnPolicyRunner


def export_policy(env, train_cfg, log_dir, ckpt, device, output_path=None):
    """Export an rsl-rl actor as a TorchScript module for LibTorch."""
    log_dir = Path(log_dir)
    print(f"Loading: {log_dir}")

    # runner
    runner = TimedOnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    resume_path = log_dir / f"model_{ckpt}.pt"
    runner.load(resume_path)

    if output_path is None:
        output_path = log_dir / "policy_traced.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(output_path.parent), filename=output_path.name)
    print(f"Saved traced policy to: {output_path}")
    return output_path
