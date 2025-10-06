# Supports
- Ubuntu20.04
# Installation
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
   git clone https://github.com/Genesis-Embodied-AI/Genesis.git -b v0.3.0
   ```
1. install required pip packages
   ```
   pip install -r $(rospack find genesis_tools)/requirements_Ubuntu20.04_agent-system_cpu.txt # use _cuda.txt if you use GPU
   ```
1. install Genesis
   ```
   cd ~/genesis_ws/Genesis
   pip install -e .
   ```
1. clone and build genesis_tools
   ```
   # change 'catkin_workspace' to your catkin workspace directory
   cd catkin_workspace/src
   git clone git@github.com:kindsenior/genesis_tools.git
   source catkin_workspace/devel/setup.bash
   deactivate # Deactivate virtual envs during catkin build
   catkin build genesis_tools
   ```
1. execute samples
   ```
   source catkin_workspace/devel/setup.bash
   source ~/genesis_ws/venv_genesis/bin/activate
   # training
   rosrun genesis_tools go2_train.py -l logs/go2_locomotion/test
   # inference
   rosrun genesis_tools go2_eval.py -l logs/go2_locomotion/test --ckpt 100
   ```
