cmake_minimum_required(VERSION 3.24)
project(image_analysis_and_classification_algorithms)

set(CMAKE_CXX_STANDARD 17)

add_executable(image_analysis_and_classification_algorithms main.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
<<<<<<<< HEAD:howework-1/CMakeLists.txt
target_link_libraries(image_analysis_and_classification_algorithms ${OpenCV_LIBS} )
========
target_link_libraries(image_analysis_and_classification_algorithms ${OpenCV_LIBS} )
>>>>>>>> homework-4:homework-4/CMakeLists.txt
