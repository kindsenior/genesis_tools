cmake_minimum_required(VERSION 3.0.2)
project(genesis_tools)

find_package(catkin REQUIRED)

# Python modules
catkin_python_setup()

catkin_package()

# Preserve the env-based shebang: catkin is configured with system Python 3.8,
# while these programs run in the activated Genesis Python 3.10 virtualenv.
install(
  PROGRAMS
  scripts/export_cfgs_to_yaml.py
  scripts/go2_locomotion/go2_backflip.py
  scripts/go2_locomotion/go2_eval.py
  scripts/go2_locomotion/go2_export.py
  scripts/go2_locomotion/go2_test.py
  scripts/go2_locomotion/go2_train.py
  scripts/sample_bipedal_robot_locomotion/sbr1_eval.py
  scripts/sample_bipedal_robot_locomotion/sbr1_export.py
  scripts/sample_bipedal_robot_locomotion/sbr1_train.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
