#include <opencv2/opencv.hpp>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include "falldetection_pipeline.hpp"



// 类型定义
typedef FalldetectionPipeline* (*CreatePipelineFunc)(const std::string&, int);
typedef void (*DestroyPipelineFunc)(FalldetectionPipeline*);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <视频路径>" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];
    std::string config_path = "models.yaml";
    int dev_id = 0;

    // 加载动态库
    void* handle = dlopen("/path/to/lib/libfalldetection.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载库: " << dlerror() << std::endl;
        return 1;
    }

    // 解析符号
    CreatePipelineFunc create_pipeline = (CreatePipelineFunc)dlsym(handle, "create_falldetection_pipeline");
    DestroyPipelineFunc destroy_pipeline = (DestroyPipelineFunc)dlsym(handle, "destroy_falldetection_pipeline");
    if (!create_pipeline || !destroy_pipeline) {
        std::cerr << "无法解析符号: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    try {
        // 创建 pipeline 实例
        FalldetectionPipeline* pipeline = create_pipeline(config_path, dev_id);
        if (!pipeline) {
            throw std::runtime_error("无法创建 FalldetectionPipeline");
        }

        // 打开视频
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开视频: " + video_path);
        }

        // 初始化输出视频
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter out("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
        if (!out.isOpened()) {
            throw std::runtime_error("无法打开输出视频: output.avi");
        }

        int frame_count = 0;
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::cout << "视频处理完成，共 " << frame_count << " 帧" << std::endl;
                break;
            }

            // 推理
            ActionInferenceResult result = pipeline->inference(frame);
            frame_count++;

            // 打印结果
            std::cout << "帧 " << frame_count << ": 检测到 " << result.online_targets.size() << " 个目标" << std::endl;
            for (size_t i = 0; i < result.labels.size(); ++i) {
                std::cout << "  目标 " << i << ": 标签=" << result.labels[i]
                    << ", 概率=" << result.probs[i]
                    << ", 关键点数=" << result.humans[i].size() << std::endl;
            }

            // 保存可视化帧
            if (!result.visualized_frame.empty()) {
                out.write(result.visualized_frame);
            }
        }

        // 释放资源
        cap.release();
        out.release();
        pipeline->reset();
        destroy_pipeline(pipeline);
        dlclose(handle);
        std::cout << "处理完成" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        dlclose(handle);
        return 1;
    }

    return 0;
}