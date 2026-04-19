# Supports
- Ubuntu20.04
- python3.10
# Installation
1. clone and build genesis_tools
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
1. install required pip packages
   ```
   source <catkin_workspace>/devel/setup.bash
   roscd genesis_tools
   pip install -r requirements_Ubuntu20.04_cpu.txt # use _gpu.txt if you use GPU
   ```

# Samples
1. execute samples
   ```
   source <catkin_workspace>/devel/setup.bash
   source ~/genesis_ws/venv_genesis/bin/activate
   # training
   rosrun genesis_tools go2_train.py -l logs/go2_locomotion/test
   # inference
   rosrun genesis_tools go2_eval.py -l logs/go2_locomotion/test --ckpt 100
   ```
