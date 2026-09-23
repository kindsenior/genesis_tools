# genesis_tools

## Supported environment

- Ubuntu 20.04
- Python 3.10
- Genesis 1.3.3
- rsl-rl-lib 2.2.4

The legacy `rsl-rl-lib==2.2.4` runner, policy configuration, and tensor
observation interface are intentionally retained.

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
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib"))'
   ```

   The expected versions are `genesis-world 1.3.3` and
   `rsl-rl-lib 2.2.4`.

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
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib"))'
   ```

   The expected versions are `genesis-world 1.3.3` and
   `rsl-rl-lib 2.2.4`.

## Samples

Relative log paths are resolved from the current working directory. Use the
same log directory for training, evaluation, and export.

### Go2

#### Catkin entry points

```bash
source <catkin_workspace>/devel/setup.bash
source ~/genesis_ws/venv_genesis/bin/activate

# Training
rosrun genesis_tools go2_train.py -l logs/go2_locomotion/test

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
  -l logs/go2_locomotion/test
python -m genesis_tools.examples.go2.eval \
  -l logs/go2_locomotion/test --ckpt 100
python -m genesis_tools.examples.go2.export \
  -l logs/go2_locomotion/test --ckpt 100
```

Both export paths create `policy_traced.pt` for loading from LibTorch. Training,
evaluation, and export use `cfgs.yaml` as the canonical configuration artifact.

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
