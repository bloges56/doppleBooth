# My system is Ubuntu 20.04 and my editor is VScode and this cmake file is built for it please modify this file to use on your own systems 
cmake_minimum_required(VERSION 3.5)

# set project name to same as current directory
get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} LANGUAGES CXX)

# # can be used to fetch dependencies
include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
)
FetchContent_MakeAvailable(dlib)

# vscode
include(CTest)
enable_testing()

# find OpenCV and dlib on current computer
find_package(OpenCV REQUIRED)
# find_package(dlib REQUIRED)

# set cmak standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add files to project
add_executable(${ProjectId} 
    main.cpp
)

# link libarys to project
target_link_libraries(${ProjectId} ${OpenCV_LIBS})
target_link_libraries(${ProjectId} dlib)

# vscode
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)