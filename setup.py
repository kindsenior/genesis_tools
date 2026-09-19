from pathlib import Path

from setuptools import find_packages, setup


packages = find_packages("src")
setup_args = {
    "name": "genesis_tools",
    "version": "0.1.0",
    "description": "Genesis tools",
    "packages": packages,
    "package_dir": {"": "src"},
}

# Reuse package.xml metadata when catkin_pkg is available, while keeping the
# project installable in a Python-only virtual environment.
try:
    from catkin_pkg.python_setup import generate_distutils_setup
except ImportError:
    pass
else:
    setup_args.update(
        generate_distutils_setup(
            packages=packages,
            package_dir={"": "src"},
        )
    )

# Install model assets for non-editable Python installations. Editable and
# catkin devel-space use can still access the files directly from the checkout.
model_root = Path(__file__).parent / "models"
setup_args["data_files"] = [
    (
        str(
            Path("share")
            / "genesis_tools"
            / "models"
            / path.parent.relative_to(model_root)
        ),
        [str(path)],
    )
    for path in model_root.rglob("*")
    if path.is_file()
]

setup(**setup_args)
