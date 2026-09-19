"""Locate model assets in source, editable, wheel, and catkin installs."""

import sysconfig
from pathlib import Path


def get_model_path(relative_path):
    """Return an existing model asset path for the active installation."""
    relative_path = Path(relative_path)
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[2] / "models" / relative_path,
        module_path.parents[1] / "share" / "genesis_tools" / "models" / relative_path,
        Path(sysconfig.get_path("data"))
        / "share"
        / "genesis_tools"
        / "models"
        / relative_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find genesis_tools model asset {relative_path}; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )
