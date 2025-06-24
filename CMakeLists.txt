cmake_minimum_required(VERSION 3.16)
project(action_recognition C CXX)

# 设置编译类型为 Debug，优化级别为 O3

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
SET(CMAKE_BUILD_TYPE "Debug")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message(STATUS "Current directory: ${PROJECT_SOURCE_DIR}")

# 默认目标架构为 pcie
if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH soc)
endif()

if(NOT DEFINED SDK)
    set(SDK /opt/sophon) # 设置正确的 Sophgo SDK 路径
endif()



function(glob_recursive output_var)
    set(all_files "")
    foreach(pattern ${ARGN}) 
        file(GLOB files "${pattern}")
        list(APPEND all_files ${files})
        # 递归查找子目录
        get_filename_component(dir "${pattern}" DIRECTORY)
        file(GLOB subdirs RELATIVE "${dir}" "${dir}/*")
        foreach(subdir ${subdirs})
            if(IS_DIRECTORY "${dir}/${subdir}")
                glob_recursive(sub_files "${dir}/${subdir}/*.cpp")
                list(APPEND all_files ${sub_files})
            endif()
        endforeach()
    endforeach()
    set(${output_var} ${all_files} PARENT_SCOPE)
endfunction()

# 函数：检查并转换为绝对路径
function(check_and_replace_path ARG_NAME) 
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        message(STATUS "Path ${PATH_TO_CHECK} exists")
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)
        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path: ${PATH_TO_CHECK}")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

# 函数：递归包含子目录
function(include_sub_directories_recursively root_dir)
    if (${root_dir} MATCHES "${CMAKE_SOURCE_DIR}/build")
        message(STATUS "Skipping include dir: ${root_dir}")
        return()
    endif()
    if (IS_DIRECTORY ${root_dir})
        message(STATUS "Including dir: ${root_dir}")
        include_directories(${root_dir})
    endif()
    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            include_sub_directories_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

# 根据目标架构配置
if (${TARGET_ARCH} STREQUAL "pcie")
    # 配置 FFmpeg
    set(FFMPEG_DIR /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    # 配置 OpenCV
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # 配置 Sophgo 库
    find_package(libsophon REQUIRED)
    find_package(yaml-cpp REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # 包含项目头文件
    include_directories(${CMAKE_SOURCE_DIR}/bytetrack_opencv/thirdparty/include)
    include_directories(${CMAKE_SOURCE_DIR}/bytetrack_opencv)
    include_directories(${CMAKE_SOURCE_DIR}/dependencies/include)
    include_directories(${CMAKE_SOURCE_DIR}/hrnet_pose_bmcv)
    include_directories(${CMAKE_SOURCE_DIR}/action_recognition)
    # include_directories(${CMAKE_SOURCE_DIR}/dependencies/include/yaml-cpp)
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})

    # 收集源文件
    glob_recursive(SRC_FILES 
        "${CMAKE_SOURCE_DIR}/bytetrack_opencv/thirdparty/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/bytetrack_opencv/*.cpp"
        "${CMAKE_SOURCE_DIR}/dependencies/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/hrnet_pose_bmcv/hrnet_pose.cpp"
        "${CMAKE_SOURCE_DIR}/action_recognition/pipeline.cpp"
    )

    # 生成可执行文件
    add_executable(action_recognition_pipeline.pcie ${SRC_FILES})
    target_link_libraries(action_recognition_pipeline.pcie 
        ${FFMPEG_LIBS} 
        ${OpenCV_LIBS} 
        ${LIBSOPHON_LIBRARIES} 
        -lpthread
        -yaml-cpp
    )

elseif (${TARGET_ARCH} STREQUAL "soc")
    # 检查 SDK 路径
    check_and_replace_path(SDK)
    message(STATUS "SDK: ${SDK}")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

    # Sophgo 库
    set(BM_LIBS bmlib bmrt bmcv yuv)


    set(FFMPEG_DIR /opt/sophon/sophon-ffmpeg-latest/lib/cmake CACHE STRING "") 
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4/ CACHE STRING "")

    #set(SN_INCLUDE "/opt/sophon/libsophon-current/include/"
    #${OpenCV_INCLUDE_DIRS}
    #${FFMPEG_INCLUDE_DIRS})

    find_package(OpenCV REQUIRED) 
    find_package(FFMPEG REQUIRED)
    find_package(yaml-cpp REQUIRED)
    # OpenCV 库
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio opencv_video)

    #link_directories(${FFMPEG_LIB_DIRS})
    #link_directories("/opt/sophon/libsophon-current/lib")
    # 查找 OpenCV
    #set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest)
    #find_package(OpenCV REQUIRED)
    #include_directories(${OpenCV_INCLUDE_DIRS})
    #link_directories(${OpenCV_LIB_DIRS})

    # 包含项目头文件
    include_directories(${CMAKE_SOURCE_DIR}/bytetrack_opencv/thirdparty/include)
    include_directories(${CMAKE_SOURCE_DIR}/bytetrack_opencv)
    include_directories(${CMAKE_SOURCE_DIR}/dependencies/include)
    include_directories(${CMAKE_SOURCE_DIR}/hrnet_pose_bmcv)
    include_directories(${CMAKE_SOURCE_DIR}/action_recognition)
    # include_directories(${SN_INCLUDE})
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})

    # 包含 SDK 头文件
    include_directories("${SDK}/libsophon-current/include")
    include_directories("${SDK}/sophon-ffmpeg-latest/include")
    include_directories("${SDK}/sophon-opencv-latest/include/opencv4")
    link_directories("${SDK}/sophon-opencv-latest/lib")
    link_directories("${SDK}/sophon-ffmpeg-latest/lib")
    link_directories("${SDK}/libsophon-current/lib")
   
    # 收集源文件
    glob_recursive(SRC_FILES 
        "${CMAKE_SOURCE_DIR}/bytetrack_opencv/thirdparty/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/bytetrack_opencv/*.cpp"
        "${CMAKE_SOURCE_DIR}/dependencies/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/hrnet_pose_bmcv/*.cpp"
        "${CMAKE_SOURCE_DIR}/action_recognition/*.cpp"
    )

    # 生成动态库
    add_library(action_recognition SHARED ${SRC_FILES})
    target_link_libraries(action_recognition
        ${OpenCV_LIBS}
        yaml-cpp
        -L${SOPHON_LIB_DIR} -lbmvideo -lbmcv -lbmlib
    )

    # 设置符号可见性
    set_target_properties(action_recognition PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
     
    # 生成可执行文件
    add_executable(action_recognition_pipeline.soc ${SRC_FILES} "${CMAKE_SOURCE_DIR}/action_recognition/test_falldetection_pipeline.cpp")
    target_link_libraries(action_recognition_pipeline.soc 
        ${BM_LIBS} 
        ${OPENCV_LIBS} 
        -lpthread 
        -lavcodec 
        -lavformat  
        -lavutil
        yaml-cpp
        dl
    )

else()
    message(FATAL_ERROR "不支持的架构，需为 soc 或 pcie，当前: ${TARGET_ARCH}")
endif()
