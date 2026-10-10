#!/usr/bin/env -S python3 -i

import argparse
from pathlib import Path

import torch

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat, quat_to_R

from genesis_tools.config import load_cfgs_yaml
from genesis_tools.examples.sbr1.env import Sbr1Env
from genesis_tools.rsl import TensorEnvAdapter, TimedOnPolicyRunner, require_rsl_rl_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="logs/sbr1_locomotion/test")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--ckpt", type=int, required=True)
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
    # reward_cfg["reward_scales"] = {}

    global env
    env = Sbr1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        rendered_envs_idx=list(range(args.num_envs))
    )

    rsl_env = TensorEnvAdapter(env)
    runner = TimedOnPolicyRunner(rsl_env, train_cfg, log_dir, device=gs.device)
    resume_path = Path(log_dir)/f"model_{args.ckpt}.pt"
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = rsl_env.reset()
    with torch.no_grad():
        i = 0
        while args.max_steps is None or i < args.max_steps:
            actions = policy(obs)
            # old
            # obs, _, rews, dones, infos = env.step(actions)
            # new
            obs, rews, dones, infos = rsl_env.step(actions)
            i += 1
            if i % 10 == 0:
                # contact = torch.norm(env.feet_contact_force, dim=2) > 1
                # contact_feet_vel = torch.square(env.feet_lin_vel) * contact.unsqueeze(-1)  # [num_envs, 2, 3]
                # contact_feet_ang = torch.square(env.feet_ang_vel) * contact.unsqueeze(-1)  # [num_envs, 2, 3]
                # penalize = torch.square(contact_feet_vel) + torch.square(contact_feet_ang)  # [num_envs, 2, 3]
                # reward = torch.sum(penalize, dim=(1, 2))
                # print(contact)
                # print(contact_feet_vel)
                # print(reward)

                # max_tq = torch.max(torch.abs(env.robot.get_dofs_control_force()))
                # max_dq = torch.max(torch.abs(env.robot.get_dofs_velocity()))
                # print(f"max_torque: {max_tq.item():.3f}, max_vel: {max_dq.item():.3f}")

                R = quat_to_R(
                    transform_quat_by_quat(torch.ones_like(env.feet_quat) * env.inv_base_init_quat, env.feet_quat)
                )
                print(R[:, :, 2, 2])



if __name__ == "__main__":
    main()

"""
# evaluation
python sbr1_eval.py -l logs/sbr1_locomotion/test --ckpt 100
"""
