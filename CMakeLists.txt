cmake_minimum_required(VERSION 3.0.2)
project(genesis_tools)

find_package(catkin REQUIRED)

# Python modules
# catkin_python_setup()

catkin_package()

# Install Python libraries for releasse
# install(
#   DIRECTORY src/genesis_tools
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
# )

# Install python scripts
# file(GLOB_RECURSE PYTHON_SCRIPTS
#   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#   scripts/*.py
# )

# catkin_install_python(
#   PROGRAMS ${PYTHON_SCRIPTS}
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
