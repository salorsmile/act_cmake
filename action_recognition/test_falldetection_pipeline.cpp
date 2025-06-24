// test_falldetection.c
#include "falldetection_handle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <opencv2/opencv2.hpp>

// 函数指针定义
typedef FalldetectionHandle(*CreateFunc)(const char*, int);
typedef void (*DestroyFunc)(FalldetectionHandle);
typedef int (*InferenceFunc)(FalldetectionHandle, const unsigned char*, int, int, int, CActionInferenceResult*);
typedef void (*FreeResultFunc)(CActionInferenceResult*);
typedef void (*ResetFunc)(FalldetectionHandle);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <视频路径或图像路径> [设备ID]\n", argv[0]);
        return 1;
    }

    // 解析命令行参数
    const char* input_path = argv[1];
    int dev_id = (argc > 2) ? atoi(argv[2]) : 0;

    // 动态加载共享库
    void* lib_handle = dlopen("libfalldetection.so", RTLD_LAZY);
    if (!lib_handle) {
        fprintf(stderr, "无法加载共享库: %s\n", dlerror());
        return 1;
    }

    // 加载函数
    CreateFunc create = (CreateFunc)dlsym(lib_handle, "falldetection_create");
    DestroyFunc destroy = (DestroyFunc)dlsym(lib_handle, "falldetection_destroy");
    InferenceFunc inference = (InferenceFunc)dlsym(lib_handle, "falldetection_inference");
    FreeResultFunc free_result = (FreeResultFunc)dlsym(lib_handle, "falldetection_free_result");
    ResetFunc reset = (ResetFunc)dlsym(lib_handle, "falldetection_reset");

    if (!create || !destroy || !inference || !free_result || !reset) {
        fprintf(stderr, "无法加载函数: %s\n", dlerror());
        dlclose(lib_handle);
        return 1;
    }

    // 创建句柄
    FalldetectionHandle handle = create("../models.yaml", dev_id);
    if (!handle) {
        fprintf(stderr, "创建 FalldetectionHandle 失败\n");
        dlclose(lib_handle);
        return 1;
    }

    // 打开视频
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "无法打开视频文件: %s\n", input_path);
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }

    // 获取视频属性
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (width <= 0 || height <= 0 || fps <= 0) {
        fprintf(stderr, "无效的视频属性: width=%d, height=%d, fps=%.2f\n", width, height, fps);
        cap.release();
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }

    // 初始化输出视频
    char output_path[256];
    snprintf(output_path, sizeof(output_path), "output_%s", strrchr(input_path, '/') ? strrchr(input_path, '/') + 1 : input_path);
    cv::VideoWriter out(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
    if (!out.isOpened()) {
        fprintf(stderr, "无法打开输出视频文件: %s\n", output_path);
        cap.release();
        destroy(handle);
        dlclose(lib_handle);
        return 1;
    }
    printf("保存视频到: %s\n", output_path);

    // 逐帧处理
    int frame_count = 0;
    while (1) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            printf("视频处理完成，共 %d 帧\n", frame_count);
            break;
        }
        if (frame.empty()) {
            fprintf(stderr, "读取到空帧，跳过\n");
            continue;
        }

        // 确保帧格式为 BGR
        if (frame.type() != CV_8UC3) {
            cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
            if (frame.type() != CV_8UC3) {
                fprintf(stderr, "帧格式转换失败，类型: %d\n", frame.type());
                continue;
            }
        }

        // 执行推理
        CActionInferenceResult result;
        memset(&result, 0, sizeof(CActionInferenceResult));
        int ret = inference(handle, frame.data, frame.cols, frame.rows, frame.channels(), &result);
        if (ret == 0) {
            // 打印推理结果
            printf("帧 %d: 检测到 %d 个目标\n", frame_count, result.online_targets.target_count);
            for (int i = 0; i < result.label_count; ++i) {
                printf("  目标 %d: 标签=%s, 概率=%.2f, 关键点数=%d\n",
                    result.online_targets.targets[i].track_id,
                    result.labels[i],
                    result.probs[i],
                    result.humans[i].point_count);
            }

            // 保存可视化帧
            if (result.visualized_frame_data) {
                cv::Mat vis_frame(result.frame_height, result.frame_width, CV_8UC3, result.visualized_frame_data);
                out.write(vis_frame);
            }

            // 释放结果
            free_result(&result);
        }
        else {
            fprintf(stderr, "帧 %d 推理失败: %d\n", frame_count, ret);
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