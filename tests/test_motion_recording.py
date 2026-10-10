import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from genesis_tools.motion_recording import (
    MotionRecorder,
    effective_dynamics_hash,
)


EFFECTIVE_DYNAMICS = {
    "schema_version": 1,
    "robot_name": "test",
    "model_hash": "a" * 64,
    "links": [],
    "joints": [],
}


class MotionRecorderTest(unittest.TestCase):
    def test_writes_reset_separated_pickle_free_sequences(self):
        recorder = MotionRecorder(
            0.1,
            static_data={
                "joint_names": np.asarray(["hip", "knee"]),
                "contact_link_names": np.asarray(["left", "right"]),
            },
        )
        for offset in (0.0, 1.0):
            for step in range(3):
                recorder.append({
                    "joint_position": np.asarray([offset, step]),
                    "joint_velocity": np.asarray([step, step + 1.0]),
                    "contact_active": np.asarray([True, False]),
                })
            recorder.finish_sequence(("reset",))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = recorder.write(
                output,
                {"robot_name": "test"},
                effective_dynamics=EFFECTIVE_DYNAMICS,
            )

            self.assertEqual(len(manifest["sequences"]), 2)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["sequence_schema_version"], 1)
            self.assertEqual(
                manifest["effective_dynamics"]["content_sha256"],
                effective_dynamics_hash(EFFECTIVE_DYNAMICS),
            )
            with np.load(output / "sequences" / "sequence_000.npz") as data:
                self.assertFalse(any(
                    value.dtype == object for value in data.values()
                ))
                np.testing.assert_allclose(data["time"], [0.0, 0.1, 0.2])
                self.assertEqual(data["joint_acceleration"].shape, (3, 2))
                self.assertEqual(int(data["schema_version"]), 1)
            with (output / "effective_dynamics.yaml").open() as stream:
                self.assertEqual(yaml.safe_load(stream), EFFECTIVE_DYNAMICS)
            with (output / "motion_export.yaml").open() as stream:
                saved = yaml.safe_load(stream)
            self.assertEqual(saved["robot_name"], "test")

    def test_rejects_shape_changes_and_drops_short_sequences(self):
        recorder = MotionRecorder(0.01, minimum_samples=2)
        recorder.append({"position": np.zeros(2)})
        self.assertFalse(recorder.finish_sequence(("reset",)))
        self.assertEqual(recorder.dropped_sequences, 1)

        recorder.append({"position": np.zeros(2)})
        with self.assertRaisesRegex(ValueError, "fields or shapes changed"):
            recorder.append({"position": np.zeros(3)})

    def test_rejects_object_arrays(self):
        with self.assertRaisesRegex(TypeError, "object arrays"):
            MotionRecorder(0.01, static_data={"bad": np.asarray([{}])})


if __name__ == "__main__":
    unittest.main()
