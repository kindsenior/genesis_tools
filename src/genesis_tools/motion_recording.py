"""Write reset-separated, pickle-free motion sequence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


MOTION_BUNDLE_SCHEMA_VERSION = 1
MOTION_SEQUENCE_SCHEMA_VERSION = 1
EFFECTIVE_DYNAMICS_SCHEMA_VERSION = 1


def effective_dynamics_hash(value):
    """Return a deterministic hash of an effective-dynamics mapping."""
    encoded = json.dumps(
        _canonical_hash_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_hash_value(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical_hash_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_effective_dynamics(output_dir, value):
    if not isinstance(value, dict):
        raise TypeError("effective_dynamics must be a mapping")
    if int(value.get("schema_version", -1)) != (
        EFFECTIVE_DYNAMICS_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported effective dynamics schema: "
            f"{value.get('schema_version')!r}"
        )
    path = Path(output_dir) / "effective_dynamics.yaml"
    with path.open("w") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
    return {
        "file": path.name,
        "schema_version": EFFECTIVE_DYNAMICS_SCHEMA_VERSION,
        "content_sha256": effective_dynamics_hash(value),
        "file_sha256": _sha256_file(path),
    }


@dataclass
class MotionSequence:
    """One continuous sequence that does not cross an environment reset."""

    samples: dict
    termination_reasons: tuple


class MotionRecorder:
    """Collect samples and split them whenever the environment resets."""

    def __init__(self, dt, *, static_data=None, minimum_samples=2):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if minimum_samples < 1:
            raise ValueError("minimum_samples must be at least one")
        self.dt = float(dt)
        self.static_data = {
            key: self._safe_array(value)
            for key, value in (static_data or {}).items()
        }
        self.minimum_samples = int(minimum_samples)
        self.sequences = []
        self.dropped_sequences = 0
        self._current = {}
        self._sample_shapes = None

    @staticmethod
    def _safe_array(value):
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError("object arrays are not allowed in motion bundles")
        return array

    def append(self, sample):
        """Append one state/action sample to the current sequence."""
        arrays = {
            key: self._safe_array(value).copy()
            for key, value in sample.items()
        }
        shapes = {key: value.shape for key, value in arrays.items()}
        if self._sample_shapes is None:
            self._sample_shapes = shapes
        elif shapes != self._sample_shapes:
            raise ValueError(
                "motion sample fields or shapes changed: "
                f"expected {self._sample_shapes}, found {shapes}"
            )
        for key, value in arrays.items():
            self._current.setdefault(key, []).append(value)

    def finish_sequence(self, termination_reasons=()):
        """Finish the current sequence, dropping it only when too short."""
        if not self._current:
            return False
        sample_count = len(next(iter(self._current.values())))
        if sample_count < self.minimum_samples:
            self.dropped_sequences += 1
            self._current = {}
            return False
        stacked = {
            key: np.stack(values, axis=0)
            for key, values in self._current.items()
        }
        if "joint_velocity" in stacked:
            velocity = stacked["joint_velocity"]
            stacked["joint_acceleration"] = (
                np.gradient(velocity, self.dt, axis=0)
                if sample_count > 1 else np.zeros_like(velocity)
            )
        self.sequences.append(MotionSequence(
            samples=stacked,
            termination_reasons=tuple(termination_reasons),
        ))
        self._current = {}
        return True

    def write(self, output_dir, bundle_metadata, *, effective_dynamics):
        """Write all completed sequences and return their manifest entries."""
        if self._current:
            self.finish_sequence(("recording_end",))
        if not self.sequences:
            raise ValueError("recording produced no sequence with enough samples")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamics_entry = _write_effective_dynamics(
            output_dir, effective_dynamics,
        )
        sequence_dir = output_dir / "sequences"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        sequence_entries = []
        for index, sequence in enumerate(self.sequences):
            stem = f"sequence_{index:03d}"
            sample_count = len(next(iter(sequence.samples.values())))
            time = np.arange(sample_count, dtype=np.float64) * self.dt
            arrays = {
                "schema_version": np.asarray(MOTION_SEQUENCE_SCHEMA_VERSION),
                "time": time,
                "dt": np.asarray(self.dt, dtype=np.float64),
                **self.static_data,
                **sequence.samples,
            }
            npz_path = sequence_dir / f"{stem}.npz"
            np.savez_compressed(npz_path, **arrays)
            entry = {
                "file": str(npz_path.relative_to(output_dir)),
                "samples": sample_count,
                "duration": float(time[-1]) if sample_count else 0.0,
                "termination_reasons": list(sequence.termination_reasons),
            }
            with (sequence_dir / f"{stem}.yaml").open("w") as stream:
                yaml.safe_dump(entry, stream, sort_keys=False)
            sequence_entries.append(entry)

        manifest = {
            "schema_version": MOTION_BUNDLE_SCHEMA_VERSION,
            "sequence_schema_version": MOTION_SEQUENCE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **bundle_metadata,
            "effective_dynamics": dynamics_entry,
            "sequences": sequence_entries,
            "dropped_short_sequences": self.dropped_sequences,
        }
        with (output_dir / "motion_export.yaml").open("w") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
        return manifest
