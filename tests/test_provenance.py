import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from genesis_tools.provenance import (
    capture_git_provenance,
    create_model_snapshot,
    find_git_root,
    inspect_model_source,
    verify_model_source,
)


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Test User")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_model(self, mass="1.0"):
        model_root = self.repo / "models" / "robot"
        (model_root / "urdf").mkdir(parents=True, exist_ok=True)
        (model_root / "meshes").mkdir(parents=True, exist_ok=True)
        (model_root / "meshes" / "body.stl").write_bytes(b"mesh-v1")
        (model_root / "urdf" / "robot.urdf").write_text(
            "<robot name=\"test\">\n"
            "  <link name=\"body\">\n"
            f"    <inertial><mass value=\"{mass}\"/></inertial>\n"
            "    <visual><geometry>"
            "<mesh filename=\"../meshes/body.stl\"/>"
            "</geometry></visual>\n"
            "  </link>\n"
            "</robot>\n"
        )
        return model_root

    def test_find_git_root_accepts_a_file(self):
        source_file = self.repo / "module.py"
        source_file.write_text("value = 1\n")

        self.assertEqual(find_git_root(source_file), self.repo)
        self.assertEqual(find_git_root(self.repo), self.repo)

    def test_git_patch_includes_untracked_text(self):
        tracked = self.repo / "tracked.py"
        tracked.write_text("value = 1\n")
        self._git("add", "tracked.py")
        self._git("commit", "-m", "initial")
        tracked.write_text("value = 2\n")
        (self.repo / "new.yaml").write_text("enabled: true\n")
        (self.repo / "mesh.bin").write_bytes(b"binary")

        output = self.root / "provenance"
        metadata = capture_git_provenance(self.repo, output)
        patch = (output / "changes_from_git_head.text.patch").read_text()

        self.assertTrue(metadata["dirty"])
        self.assertIn("-value = 1", patch)
        self.assertIn("+value = 2", patch)
        self.assertIn("b/new.yaml", patch)
        self.assertIn("+enabled: true", patch)
        self.assertNotIn("mesh.bin", patch)
        untracked = (output / "untracked_files.txt").read_text()
        self.assertIn("mesh.bin", untracked)

    def test_source_identity_can_be_verified_without_using_snapshot(self):
        model_root = self._write_model(mass="1.0")
        urdf = model_root / "urdf" / "robot.urdf"
        options = {"convexify": True}
        identity = inspect_model_source(
            source_root=model_root,
            urdf_path=urdf,
            build_options=options,
        )

        verified = verify_model_source(
            identity["model_hash"],
            source_root=model_root,
            urdf_path=urdf,
            build_options=options,
        )
        self.assertEqual(identity["model_hash"], verified["model_hash"])

        self._write_model(mass="2.0")
        with self.assertRaisesRegex(ValueError, "robot model changed"):
            verify_model_source(
                identity["model_hash"],
                source_root=model_root,
                urdf_path=urdf,
                build_options=options,
            )

    def test_model_snapshot_hashes_urdf_assets_and_build_options(self):
        model_root = self._write_model()
        self._git("add", ".")
        self._git("commit", "-m", "model")
        artifacts = self.root / "artifacts"

        first_dir, first = create_model_snapshot(
            source_root=model_root,
            urdf_path=model_root / "urdf" / "robot.urdf",
            artifact_root=artifacts,
            robot_name="test_robot",
            build_options={"convexify": True},
        )
        reused_dir, reused = create_model_snapshot(
            source_root=model_root,
            urdf_path=model_root / "urdf" / "robot.urdf",
            artifact_root=artifacts,
            robot_name="test_robot",
            build_options={"convexify": True},
        )

        self.assertEqual(first_dir, reused_dir)
        self.assertEqual(first["model_hash"], reused["model_hash"])
        self.assertTrue(
            (first_dir / "robot_model" / "urdf" / "robot.urdf").is_file()
        )
        self.assertTrue(
            (first_dir / "robot_model" / "meshes" / "body.stl").is_file()
        )
        with (first_dir / "model.yaml").open() as stream:
            saved = yaml.safe_load(stream)
        self.assertEqual(saved["model_hash"], first["model_hash"])

        changed_options_dir, changed_options = create_model_snapshot(
            source_root=model_root,
            urdf_path=model_root / "urdf" / "robot.urdf",
            artifact_root=artifacts,
            robot_name="test_robot",
            build_options={"convexify": False},
        )
        self.assertNotEqual(first_dir, changed_options_dir)
        self.assertNotEqual(
            first["model_hash"], changed_options["model_hash"]
        )

        snapshot_mesh = (
            first_dir / "robot_model" / "meshes" / "body.stl"
        )
        snapshot_mesh.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "snapshot was modified"):
            create_model_snapshot(
                source_root=model_root,
                urdf_path=model_root / "urdf" / "robot.urdf",
                artifact_root=artifacts,
                robot_name="test_robot",
                build_options={"convexify": True},
            )

    def test_parent_snapshot_writes_text_diff(self):
        model_root = self._write_model(mass="1.0")
        self._git("add", ".")
        self._git("commit", "-m", "model")
        artifacts = self.root / "artifacts"
        parent_dir, parent = create_model_snapshot(
            source_root=model_root,
            urdf_path=model_root / "urdf" / "robot.urdf",
            artifact_root=artifacts,
            robot_name="test_robot",
        )

        self._write_model(mass="2.0")
        child_dir, child = create_model_snapshot(
            source_root=model_root,
            urdf_path=model_root / "urdf" / "robot.urdf",
            artifact_root=artifacts,
            robot_name="test_robot",
            parent_model_hash=parent["model_hash"][:16],
        )
        patch = (
            child_dir
            / "provenance"
            / "changes_from_parent_model.text.patch"
        ).read_text()

        self.assertNotEqual(parent_dir, child_dir)
        self.assertEqual(child["parent_model_hash"], parent["model_hash"])
        self.assertIn('mass value="1.0"', patch)
        self.assertIn('mass value="2.0"', patch)

    def test_missing_parent_does_not_create_child_directory(self):
        model_root = self._write_model()
        self._git("add", ".")
        self._git("commit", "-m", "model")
        artifacts = self.root / "artifacts"

        with self.assertRaises(FileNotFoundError):
            create_model_snapshot(
                source_root=model_root,
                urdf_path=model_root / "urdf" / "robot.urdf",
                artifact_root=artifacts,
                robot_name="test_robot",
                parent_model_hash="deadbeefdeadbeef",
            )

        robot_artifacts = artifacts / "test_robot"
        self.assertFalse(robot_artifacts.exists())


if __name__ == "__main__":
    unittest.main()
