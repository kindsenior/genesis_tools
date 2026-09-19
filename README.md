# Supports
- Ubuntu20.04
- Python 3.10
- Genesis v1.3.3
- rsl-rl-lib v2.2.4

# Installation

## Catkin

1. Clone and build `genesis_tools`.

   ```
   # change 'catkin_workspace' to your catkin workspace directory
   cd <catkin_workspace>/src
   git clone git@github.com:kindsenior/genesis_tools.git
   source <catkin_workspace>/devel/setup.bash
   deactivate # Deactivate virtual envs during catkin build, if you already activate venv
   catkin build genesis_tools
   ```
1. install python3.10
   ```
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   ```
1. generate virtual env
   ```
   mkdir ~/genesis_ws
   cd ~/genesis_ws
   python3.10 -m venv venv_genesis
   ```
1. activate virtual env
   ```
   source ~/genesis_ws/venv_genesis/bin/activate
   ```
1. get genesis sources
   ```
   git clone --branch v1.3.3 --depth 1 \
     https://github.com/Genesis-Embodied-AI/Genesis.git
   ```
1. Install required pip packages.

   ```
   source <catkin_workspace>/devel/setup.bash
   roscd genesis_tools
   pip install -r requirements_Ubuntu20.04_cpu.txt # use _gpu.txt if you use GPU
   ```
1. install Genesis
   ```
   cd ~/genesis_ws/Genesis
   pip install -e .
   ```

## Python only

The same source tree can be installed directly into an active virtual
environment without building or sourcing a catkin workspace.

```
cd <path-to-genesis_tools>
pip install -r requirements_Ubuntu20.04_cpu.txt # use _gpu.txt for GPU
pip install -e .
```

# Samples

## Catkin entry points

   ```
   source <catkin_workspace>/devel/setup.bash
   source ~/genesis_ws/venv_genesis/bin/activate
   # training
   rosrun genesis_tools go2_train.py -l logs/go2_locomotion/test
   # inference
   rosrun genesis_tools go2_eval.py -l logs/go2_locomotion/test --ckpt 100
   # LibTorch export
   rosrun genesis_tools go2_export.py -l logs/go2_locomotion/test --ckpt 100
   ```

## Python module entry points

```
python -m genesis_tools.examples.go2.train -l logs/go2_locomotion/test
python -m genesis_tools.examples.go2.eval \
  -l logs/go2_locomotion/test --ckpt 100
python -m genesis_tools.examples.go2.export \
  -l logs/go2_locomotion/test --ckpt 100
```

Both export paths create `policy_traced.pt` for loading from LibTorch. Training,
evaluation, and export use `cfgs.yaml` as the canonical configuration artifact.

## Legacy pickle conversion

`export_cfgs_to_yaml.py` is only for trusted legacy or upstream logs containing
`cfgs.pkl`. Pickle loading can execute code, so do not use it with untrusted
files.

```
rosrun genesis_tools export_cfgs_to_yaml.py --log_dir <legacy-log-directory>
# or, without catkin
python -m genesis_tools.legacy.pickle_to_yaml \
  --log_dir <legacy-log-directory>
```
