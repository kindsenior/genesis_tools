"""Create reproducible model and source-code provenance artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml


TEXT_SUFFIXES = {
    ".cfg", ".cmake", ".json", ".md", ".py", ".rst", ".sh",
    ".toml", ".txt", ".urdf", ".xacro", ".xml", ".yaml", ".yml",
}


def sha256_file(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo, *args, check=True):
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def find_git_root(path):
    """Return the containing Git root, or ``None`` outside a repository."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    try:
        root = _run_git(path, "rev-parse", "--show-toplevel").strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(root)


def _is_text_path(path):
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def _untracked_paths(repo):
    output = _run_git(
        repo, "ls-files", "--others", "--exclude-standard", "-z"
    )
    return tuple(Path(value) for value in output.split("\0") if value)


def _new_file_patch(path, relative_path):
    lines = Path(path).read_text(errors="replace").splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        [], lines,
        fromfile="/dev/null",
        tofile=f"b/{relative_path.as_posix()}",
    ))


def capture_git_provenance(repo, output_dir, *, include_paths=None):
    """Write Git metadata and a text patch including untracked text files."""
    repo = Path(repo).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_paths = tuple(Path(path) for path in (include_paths or ()))

    revision = _run_git(repo, "rev-parse", "HEAD").strip()
    branch = _run_git(repo, "branch", "--show-current").strip() or None
    status_args = ["status", "--porcelain=v1", "--untracked-files=all"]
    if include_paths:
        status_args.extend(("--", *(str(path) for path in include_paths)))
    status = _run_git(repo, *status_args)
    remote = _run_git(
        repo, "config", "--get", "remote.origin.url", check=False
    ).strip() or None

    changed_args = ["diff", "--name-only", "-z", "HEAD", "--"]
    changed_args.extend(str(path) for path in include_paths)
    changed_paths = tuple(
        Path(value)
        for value in _run_git(repo, *changed_args).split("\0")
        if value
    )
    selected_tracked = tuple(
        path for path in changed_paths if _is_text_path(path)
    )
    tracked_patch = ""
    if selected_tracked:
        tracked_patch = _run_git(
            repo, "diff", "--no-ext-diff", "--text", "HEAD", "--",
            *(str(path) for path in selected_tracked),
        )

    selected_untracked_all = []
    selected_untracked = []
    for relative_path in _untracked_paths(repo):
        if include_paths and not any(
            relative_path == candidate or candidate in relative_path.parents
            for candidate in include_paths
        ):
            continue
        selected_untracked_all.append(relative_path)
        if _is_text_path(relative_path):
            selected_untracked.append(relative_path)

    untracked_patch = "".join(
        _new_file_patch(repo / path, path) for path in selected_untracked
    )
    patch_path = output_dir / "changes_from_git_head.text.patch"
    patch_path.write_text(tracked_patch + untracked_patch)
    (output_dir / "untracked_files.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in selected_untracked_all)
    )

    metadata = {
        "repository": str(repo),
        "remote": remote,
        "branch": branch,
        "revision": revision,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_text_files": [str(path) for path in selected_tracked],
        "untracked_files": [str(path) for path in selected_untracked_all],
        "untracked_text_files": [str(path) for path in selected_untracked],
        "patch": patch_path.name,
        "patch_sha256": sha256_file(patch_path),
    }
    with (output_dir / "git.yaml").open("w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return metadata


def _model_files(source_root, urdf_path):
    source_root = Path(source_root).resolve()
    urdf_path = Path(urdf_path).resolve()
    paths = {urdf_path}
    for element in ET.parse(urdf_path).getroot().iter():
        filename = element.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            raise ValueError(
                f"package:// model resource is not supported yet: {filename}"
            )
        referenced = Path(filename)
        if not referenced.is_absolute():
            referenced = (urdf_path.parent / referenced).resolve()
        try:
            referenced.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"model resource is outside source root: {referenced}"
            ) from exc
        if not referenced.is_file():
            raise FileNotFoundError(referenced)
        paths.add(referenced)
    return tuple(sorted(
        paths, key=lambda path: path.relative_to(source_root).as_posix()
    ))


def _model_identity(source_root, files, build_options):
    source_root = Path(source_root).resolve()
    entries = []
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(source_root).as_posix()
        file_digest = sha256_file(path)
        entries.append({"path": relative_path, "sha256": file_digest})
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\0")
    options_bytes = json.dumps(
        build_options, sort_keys=True, separators=(",", ":")
    ).encode()
    digest.update(b"model_build_options\0")
    digest.update(options_bytes)
    return digest.hexdigest(), entries


def inspect_model_source(*, source_root, urdf_path, build_options=None):
    """Return the identity of a source model without copying it."""
    source_root = Path(source_root).resolve()
    urdf_path = Path(urdf_path).resolve()
    build_options = dict(build_options or {})
    files = _model_files(source_root, urdf_path)
    model_hash, file_entries = _model_identity(
        source_root, files, build_options
    )
    return {
        "model_hash": model_hash,
        "source_root": str(source_root),
        "urdf": urdf_path.relative_to(source_root).as_posix(),
        "build_options": build_options,
        "files": file_entries,
    }


def verify_model_source(
    expected_hash, *, source_root, urdf_path, build_options=None,
):
    """Raise if the current source model differs from ``expected_hash``."""
    identity = inspect_model_source(
        source_root=source_root,
        urdf_path=urdf_path,
        build_options=build_options,
    )
    if identity["model_hash"] != expected_hash:
        raise ValueError(
            "robot model changed while it was being used: expected "
            f"{expected_hash}, found {identity['model_hash']}"
        )
    return identity


def _snapshot_text_diff(parent_model_dir, model_dir):
    parent_model_dir = Path(parent_model_dir)
    model_dir = Path(model_dir)
    relative_paths = {
        path.relative_to(parent_model_dir)
        for path in parent_model_dir.rglob("*")
        if path.is_file() and _is_text_path(path)
    }
    relative_paths.update(
        path.relative_to(model_dir)
        for path in model_dir.rglob("*")
        if path.is_file() and _is_text_path(path)
    )
    chunks = []
    for relative_path in sorted(
        relative_paths, key=lambda path: path.as_posix()
    ):
        old_path = parent_model_dir / relative_path
        new_path = model_dir / relative_path
        old_lines = (
            old_path.read_text(errors="replace").splitlines(keepends=True)
            if old_path.is_file() else []
        )
        new_lines = (
            new_path.read_text(errors="replace").splitlines(keepends=True)
            if new_path.is_file() else []
        )
        chunks.extend(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=(
                f"a/{relative_path.as_posix()}"
                if old_path.is_file() else "/dev/null"
            ),
            tofile=(
                f"b/{relative_path.as_posix()}"
                if new_path.is_file() else "/dev/null"
            ),
        ))
    return "".join(chunks)


def _load_model_metadata(model_dir):
    metadata_path = Path(model_dir) / "model.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"model metadata not found: {metadata_path}")
    with metadata_path.open() as stream:
        return yaml.safe_load(stream)


def _verify_model_snapshot(model_dir, metadata):
    snapshot_dir = Path(model_dir) / "robot_model"
    for entry in metadata.get("files", ()):
        snapshot_path = snapshot_dir / entry["path"]
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"model snapshot file not found: {snapshot_path}"
            )
        actual_hash = sha256_file(snapshot_path)
        if actual_hash != entry["sha256"]:
            raise ValueError(
                f"model snapshot was modified: {snapshot_path}"
            )


def create_model_snapshot(
    *, source_root, urdf_path, artifact_root, robot_name,
    build_options=None, parent_model_hash=None,
):
    """Create or reuse a content-addressed robot-model snapshot."""
    source_root = Path(source_root).resolve()
    urdf_path = Path(urdf_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    build_options = dict(build_options or {})
    identity = inspect_model_source(
        source_root=source_root,
        urdf_path=urdf_path,
        build_options=build_options,
    )
    model_hash = identity["model_hash"]
    file_entries = identity["files"]
    model_dir = artifact_root / robot_name / f"model-{model_hash[:12]}"
    snapshot_dir = model_dir / "robot_model"
    metadata_path = model_dir / "model.yaml"

    if metadata_path.exists():
        existing = _load_model_metadata(model_dir)
        if existing.get("model_hash") != model_hash:
            raise ValueError(f"model hash mismatch in {metadata_path}")
        _verify_model_snapshot(model_dir, existing)
        return model_dir, existing
    if model_dir.exists():
        raise FileExistsError(
            f"incomplete model artifact directory already exists: {model_dir}"
        )

    parent_dir = None
    parent_metadata = None
    if parent_model_hash:
        if not all(
            character in "0123456789abcdef"
            for character in parent_model_hash
        ):
            raise ValueError("parent model hash must be lowercase hexadecimal")
        directory_prefix = parent_model_hash[:12]
        matches = sorted(
            (artifact_root / robot_name).glob(
                f"model-{directory_prefix}*"
            )
        )
        if not matches:
            raise FileNotFoundError(
                f"parent model snapshot not found: {parent_model_hash}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"parent model hash prefix is ambiguous: {parent_model_hash}"
            )
        parent_dir = matches[0]
        parent_metadata = _load_model_metadata(parent_dir)
        saved_parent_hash = parent_metadata.get("model_hash")
        if not saved_parent_hash or not saved_parent_hash.startswith(
            parent_model_hash
        ):
            raise ValueError(
                f"parent model hash does not match {parent_dir / 'model.yaml'}"
            )
        _verify_model_snapshot(parent_dir, parent_metadata)
        parent_model_hash = saved_parent_hash

    for entry in file_entries:
        source_path = source_root / entry["path"]
        relative_path = source_path.relative_to(source_root)
        destination = snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_name": robot_name,
        "model_hash": model_hash,
        "parent_model_hash": parent_model_hash,
        "urdf": str(Path("robot_model") / urdf_path.relative_to(source_root)),
        "build_options": build_options,
        "files": file_entries,
    }

    repo = find_git_root(source_root)
    if repo is not None:
        metadata["source_git"] = capture_git_provenance(
            repo,
            model_dir / "provenance",
            include_paths=(source_root.relative_to(repo),),
        )

    if parent_dir is not None:
        parent_snapshot = parent_dir / "robot_model"
        provenance_dir = model_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        (provenance_dir / "changes_from_parent_model.text.patch").write_text(
            _snapshot_text_diff(parent_snapshot, snapshot_dir)
        )
        previous_files = {
            entry["path"]: entry["sha256"]
            for entry in parent_metadata.get("files", ())
        }
        current_files = {
            entry["path"]: entry["sha256"] for entry in file_entries
        }
        metadata["changed_non_text_files"] = [
            {
                "path": relative_path,
                "previous_sha256": previous_files.get(relative_path),
                "sha256": current_files.get(relative_path),
            }
            for relative_path in sorted(set(previous_files) | set(current_files))
            if not _is_text_path(relative_path)
            and previous_files.get(relative_path) != current_files.get(relative_path)
        ]

    model_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return model_dir, metadata


def write_software_environment(path, *, extra_packages=()):
    """Write the interpreter and selected package versions to YAML."""
    packages = {}
    for package_name in (
        "genesis-world", "torch", "rsl-rl-lib", *extra_packages
    ):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "packages": packages,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return data
