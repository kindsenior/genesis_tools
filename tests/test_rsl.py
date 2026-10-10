import tempfile
import unittest
from pathlib import Path

import torch

from genesis_tools.rsl import (
    TensorEnvAdapter,
    TimedOnPolicyRunner,
    require_actor_critic_checkpoint,
    require_rsl_rl_version,
)


def training_config():
    return {
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "schedule": "fixed",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "logger": "tensorboard",
        "num_steps_per_env": 4,
        "save_interval": 1,
        "seed": 1,
    }


class FakeTensorEnv:
    num_envs = 4
    num_actions = 2
    device = "cpu"
    env_cfg = {}
    max_episode_length = 4

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.observations = torch.zeros(self.num_envs, 6)

    def get_observations(self):
        return self.observations, {"observations": {"critic": self.observations}}

    def reset(self, envs_idx=None):
        return self.get_observations()

    def step(self, actions):
        self.observations = torch.cat([actions, actions, actions], dim=1).detach()
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[dones] = 0
        extras = {
            "time_outs": dones.float(),
            "episode": {"termination_test_ratio": 1.0},
            "observations": {"critic": self.observations},
        }
        return self.observations, -actions.square().sum(1), dones, extras


class RslCompatibilityTest(unittest.TestCase):
    def test_training_checkpoint_resume_and_export(self):
        self.assertEqual(require_rsl_rl_version(), "5.5.1")
        env = TensorEnvAdapter(FakeTensorEnv())

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory)
            runner = TimedOnPolicyRunner(env, training_config(), str(log_dir), device="cpu")
            runner.learn(2, init_at_random_ep_len=True)
            checkpoint = log_dir / "model_1.pt"
            require_actor_critic_checkpoint(checkpoint)

            observations = env.get_observations()
            with torch.no_grad():
                expected = runner.get_inference_policy()(observations).clone()

            restored = TimedOnPolicyRunner(
                env,
                training_config(),
                str(log_dir / "resume"),
                device="cpu",
            )
            restored.load(checkpoint, map_location="cpu")
            with torch.no_grad():
                torch.testing.assert_close(
                    restored.get_inference_policy()(observations), expected
                )

            restored.export_policy_to_jit(str(log_dir), filename="policy.pt")
            self.assertTrue((log_dir / "policy.pt").is_file())

    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "legacy.pt"
            torch.save({"model_state_dict": {}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Legacy or unsupported"):
                require_actor_critic_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
