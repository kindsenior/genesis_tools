# genesis_tools

## Supported environment

- Ubuntu 20.04
- Python 3.10
- Genesis 1.3.3
- rsl-rl-lib 5.5.1
- PyTorch 2.11.0 (CUDA 12.8 or CPU wheel)

The examples use the rsl-rl 5.5.1 actor/critic configuration and TensorDict
observation groups. Checkpoints written by rsl-rl 2.2.4 are intentionally
rejected with an explicit compatibility error; retrain them with this stack or
evaluate them in the old environment.

PyTorch is selected for the host's CUDA/driver environment, not by rsl-rl
alone. The checked-in GPU lock is the stack validated here: PyTorch 2.11.0 with
CUDA 12.8. Use the CPU lock on a CPU-only machine; PyTorch 2.8.0 is not a
requirement of rsl-rl 5.5.1.

## Installation

There are two installation methods:

- **Catkin:** builds the ROS package and provides the `rosrun` entry points.
- **Python only:** installs the repository with `pip` and uses the Python
  module entry points without building or sourcing a catkin workspace.

Both methods use the dependency versions locked by `genesis_tools`.

### Catkin

1. Install Python 3.10.

   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   ```

2. Clone and build `genesis_tools` in a catkin workspace.

   ```bash
   cd <catkin_workspace>/src
   git clone git@github.com:kindsenior/genesis_tools.git
   cd ..
   deactivate  # Deactivate an active virtual environment before building.
   catkin build genesis_tools
   ```

3. Create and activate a virtual environment.

   ```bash
   mkdir -p ~/genesis_ws
   python3.10 -m venv ~/genesis_ws/venv_genesis
   source ~/genesis_ws/venv_genesis/bin/activate
   ```

4. Install the locked Genesis 1.3.3 stack.

   ```bash
   source <catkin_workspace>/devel/setup.bash
   roscd genesis_tools
   python -m pip install --upgrade pip
   python -m pip install -r requirements_Ubuntu20.04_gpu.txt
   # Use requirements_Ubuntu20.04_cpu.txt instead on a CPU-only machine.
   ```

5. Confirm the installed versions.

   ```bash
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib")); print("torch", version("torch"))'
   ```

   The expected versions are `genesis-world 1.3.3` and
   `rsl-rl-lib 5.5.1`, and the PyTorch variant selected by the requirements
   file (`2.11.0+cu128` or `2.11.0+cpu`).

### Python only

1. Install Python 3.10 and create a virtual environment.

   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   mkdir -p ~/genesis_ws
   python3.10 -m venv ~/genesis_ws/venv_genesis
   source ~/genesis_ws/venv_genesis/bin/activate
   ```

2. Clone the repository into any working directory.

   ```bash
   cd <work_directory>
   git clone git@github.com:kindsenior/genesis_tools.git
   ```

3. Install the locked dependencies and the Python package.

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r genesis_tools/requirements_Ubuntu20.04_gpu.txt
   # Use requirements_Ubuntu20.04_cpu.txt instead on a CPU-only machine.
   python -m pip install -e ./genesis_tools
   ```

4. Confirm the installed versions.

   ```bash
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib")); print("torch", version("torch"))'
   ```

   The expected versions are `genesis-world 1.3.3` and
   `rsl-rl-lib 5.5.1`, and the PyTorch variant selected by the requirements
   file (`2.11.0+cu128` or `2.11.0+cpu`).

## Samples

Relative log paths are resolved from the current working directory. Use the
same log directory for training, evaluation, and export.

### Go2

#### Catkin entry points

```bash
source <catkin_workspace>/devel/setup.bash
source ~/genesis_ws/venv_genesis/bin/activate

# Training: writes model_100.pt as the final checkpoint
rosrun genesis_tools go2_train.py \
  -l logs/go2_locomotion/test --max-iterations 100

# Evaluation
rosrun genesis_tools go2_eval.py \
  -l logs/go2_locomotion/test --ckpt 100

# LibTorch export
rosrun genesis_tools go2_export.py \
  -l logs/go2_locomotion/test --ckpt 100
```

#### Python module entry points

```bash
python -m genesis_tools.examples.go2.train \
  -l logs/go2_locomotion/test --max-iterations 100
python -m genesis_tools.examples.go2.eval \
  -l logs/go2_locomotion/test --ckpt 100
python -m genesis_tools.examples.go2.export \
  -l logs/go2_locomotion/test --ckpt 100
```

Both export paths create `policy_traced.pt` for loading from LibTorch. Training,
evaluation, and export use `cfgs.yaml` as the canonical configuration artifact.
An existing training directory is rejected unless `--resume` or the explicit
destructive option `--overwrite` is supplied.
`--max-iterations N` runs iterations 0 through N for a fresh run, so the
final checkpoint is `model_N.pt`. `--stop-iteration N` instead selects an
absolute global iteration and is mutually exclusive with `--max-iterations`.

### SBR1

The packaged SBR1 example uses the same shared rsl-rl boundary as Go2 while
keeping its robot-specific environment in `genesis_tools.examples.sbr1`.

```bash
# Catkin entry points
rosrun genesis_tools sbr1_train.py \
  -l logs/sbr1_locomotion/test --max-iterations 1000
rosrun genesis_tools sbr1_eval.py \
  -l logs/sbr1_locomotion/test --ckpt 100
rosrun genesis_tools sbr1_export.py \
  -l logs/sbr1_locomotion/test --ckpt 100

# Python module entry points
python -m genesis_tools.examples.sbr1.train \
  -l logs/sbr1_locomotion/test --max-iterations 1000
python -m genesis_tools.examples.sbr1.eval \
  -l logs/sbr1_locomotion/test --ckpt 100
python -m genesis_tools.examples.sbr1.export \
  -l logs/sbr1_locomotion/test --ckpt 100
```

SBR1 also writes `cfgs.yaml`, rejects accidental replacement of an existing
training directory, supports checkpoint resume, and exports
`policy_traced.pt`.

### Model and source provenance

`genesis_tools.provenance` provides shared helpers for content-addressed robot
model snapshots and training provenance. A model hash covers the URDF, every
file resource referenced by the URDF, and the Genesis model-build options.
Snapshots retain the original relative paths for archival and fallback use.
`inspect_model_source()` and `verify_model_source()` let a caller load the
original source model while checking that it still matches the archived hash.

Git provenance records the revision, remote, branch, status, all untracked file
names, and an AI-readable patch for tracked and untracked text files. Package
versions and the Python interpreter are written separately to
`software_environment.yaml`. The JAXON3 example in `genesis_tutorials` uses
these helpers with exact run paths and a shared content-addressed robot-model
store.

### Motion recording

`genesis_tools.motion_recording.MotionRecorder` is the robot-independent
writer for structural-analysis motion bundles. It accepts numeric/string
samples from a robot-specific sampler, separates sequences at explicit reset
boundaries, derives joint acceleration within each continuous sequence, and
writes compressed NPZ files that can be opened with `allow_pickle=False`.
Bundle schema 1 requires a hashed `effective_dynamics.yaml` and uses sequence
schema 1 for each time-series NPZ. Keeping the two version fields distinct lets
the motion-array format evolve independently while recording the built
simulator's link inertias, joint frames/axes, and runtime DOF parameters needed
for model reconstruction.
Robot-specific Genesis state and contact extraction, policy loading, and link
selection remain in `genesis_tutorials`; JAXON3 is the first adapter.

### Legacy pickle conversion

`export_cfgs_to_yaml.py` is only for trusted legacy or upstream logs containing
`cfgs.pkl`. Pickle loading can execute code, so do not use it with untrusted
files.

```bash
rosrun genesis_tools export_cfgs_to_yaml.py --log_dir <legacy-log-directory>
# Or, without catkin:
python -m genesis_tools.legacy.pickle_to_yaml \
  --log_dir <legacy-log-directory>
```
