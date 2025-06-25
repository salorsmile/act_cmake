// test_falldetection_pipeline.cpp
#include "falldetection_handle.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dlfcn.h>
#include <opencv2/opencv2.hpp>

// 函数指针定义
typedef FalldetectionHandle(*CreateFunc)(const char*, int);
typedef void (*DestroyFunc)(FalldetectionHandle);
typedef int (*InferenceFunc)(FalldetectionHandle, void*, CActionInferenceResult*);
typedef void (*FreeResultFunc)(CActionInferenceResult*);
typedef void (*ResetFunc)(FalldetectionHandle);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <视频路径或图像路径> [设备ID]" << std::endl;
        return 1;
    }

    // 解析命令行参数
    const char* input_path = argv[1];
    int dev_id = (argc > 2) ? std::atoi(argv[2]) : 0;

    void* lib_handle = dlopen("./libaction_recognition.so", RTLD_LAZY);
    if (!lib_handle) {
        std::cerr << "无法加载共享库: " << dlerror() << std::endl;
        return 1;
    }

    // 加载函数
    CreateFunc create = (CreateFunc)dlsym(lib_handle, "falldetection_create");
    DestroyFunc destroy = (DestroyFunc)dlsym(lib_handle, "falldetection_destroy");
    InferenceFunc inference = (InferenceFunc)dlsym(lib_handle, "falldetection_inference");
    FreeResultFunc free_result = (FreeResultFunc)dlsym(lib_handle, "falldetection_free_result");
    ResetFunc reset = (ResetFunc)dlsym(lib_handle, "falldetection_reset");

    if (!create || !destroy || !inference || !free_result || !reset) {
        std::cerr << "无法加载函数: " << dlerror() << std::endl;
        dlclose(lib_handle);
        return 1;
    }

    // 创建句柄
    FalldetectionHandle handle = create("../models.yaml", dev_id);
    if (!handle) {
        std::cerr << "创建 FalldetectionHandle 失败" << std::endl;
        dlclose(lib_handle);
        return 1;
    }

    // 打开视频
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件: " << input_path << std::endl;
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }

    // 获取视频属性
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (width <= 0 || height <= 0 || fps <= 0) {
        std::cerr << "无效的视频属性: width=" << width << ", height=" << height << ", fps=" << fps << std::endl;
        cap.release();
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }

    std::string output_path = "output_" + std::string(strrchr(input_path, '/') ? strrchr(input_path, '/') + 1 : input_path);
    cv::VideoWriter out(output_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
    if (!out.isOpened()) {
        std::cerr << "无法打开输出视频文件: " << output_path << std::endl;
        cap.release();
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }
    std::cout << "保存视频到: " << output_path << std::endl;

    // 逐帧处理
    int frame_count = 0;
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cout << "视频处理完成，共 " << frame_count << " 帧" << std::endl;
            break;
        }
        if (frame.empty()) {
            std::cerr << "读取到空帧，跳过" << std::endl;
            continue;
        }

        // 确保帧格式为 BGR
        if (frame.type() != CV_8UC3) {
            cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
            if (frame.type() != CV_8UC3) {
                std::cerr << "帧格式转换失败，类型: " << frame.type() << std::endl;
                continue;
            }
        }

        // 执行推理
        CActionInferenceResult result;
        std::memset(&result, 0, sizeof(CActionInferenceResult));
        int ret = inference(handle, &frame, &result);
        if (ret == 0) {
            std::cout << "帧 " << frame_count << ": 检测到 " << result.online_targets.target_count << " 个目标" << std::endl;
            int max_count = std::min({ result.label_count, result.human_count, result.online_targets.target_count });
            for (int i = 0; i < max_count; ++i) {
                std::cout << "  目标 " << result.online_targets.targets[i].track_id
                    << ": 标签=" << (result.labels[i] ? result.labels[i] : "N/A")
                    << ", 概率=" << result.probs[i]
                    << ", 关键点数=" << result.humans[i].point_count << std::endl;
            }

            // 保存可视化帧
            if (result.visualized_frame_data) {
                cv::Mat vis_frame(result.frame_height, result.frame_width, CV_8UC3, result.visualized_frame_data);
                std::cout << "vis_frame type: " << vis_frame.type() << ", channels: " << vis_frame.channels() << std::endl;
                if (vis_frame.type() != CV_8UC3) {
                    std::cerr << "Unexpected frame type: " << vis_frame.type() << ", converting to CV_8UC3" << std::endl;
                    cv::cvtColor(vis_frame, vis_frame, cv::COLOR_RGBA2BGR);
                    if (vis_frame.type() != CV_8UC3) {
                        std::cerr << "Format conversion failed" << std::endl;
                    }
                }
                std::string debug_image = "debug_frame_" + std::to_string(frame_count) + ".png";
                cv::imwrite(debug_image, vis_frame);
                std::cout << "保存调试图像: " << debug_image << std::endl;
                if (result.frame_width != width || result.frame_height != height) {
                    std::cerr << "帧尺寸不匹配: result=(" << result.frame_width << "," << result.frame_height
                        << "), expected=(" << width << "," << height << ")" << std::endl;
                    cv::Size output_size(result.frame_width, result.frame_height);
                    out.release();
                    out.open(output_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, output_size);
                    if (!out.isOpened()) {
                        std::cerr << "重新打开输出视频文件失败: " << output_path << std::endl;
                        cap.release();
                        destroy(handle);
                        dlclose(lib_handle);
                        return 1;
                    }
                    std::cout << "调整视频分辨率至: " << result.frame_width << "x" << result.frame_height << std::endl;
                }
                out.write(vis_frame);
            }

            // 释放结果
            free_result(&result);
        }
        else {
            std::cerr << "帧 " << frame_count << " 推理失败: " << ret << std::endl;
        }

        frame_count++;
    }

    // 清理
    out.release();
    cap.release();
    reset(handle);
    destroy(handle);
    dlclose(lib_handle);

    return 0;
}