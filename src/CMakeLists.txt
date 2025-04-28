cmake_minimum_required(VERSION 3.0.2)
project(qt_confusion_detection)

find_package(catkin REQUIRED)

catkin_metapackage()

# Install launch files
install(DIRECTORY ../launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
