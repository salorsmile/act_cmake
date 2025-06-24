
#include "falldetection_pipeline.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// 测试函数：逐帧读取视频并调用 inference
void test_inference(const std::string& config_path, const std::string& video_path, int dev_id) {
    std::cout << "开始测试 FalldetectionPipeline，视频路径: " << video_path << ", 设备ID: " << dev_id << std::endl;

    try {
        // 初始化 FalldetectionPipeline
        FalldetectionPipeline pipeline(config_path, dev_id);

        // 打开视频文件
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开视频文件: " + video_path);
        }

        // 获取视频属性
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        if (width <= 0 || height <= 0 || fps <= 0) {
            throw std::runtime_error("无效的视频属性: width=" + std::to_string(width) +
                ", height=" + std::to_string(height) +
                ", fps=" + std::to_string(fps));
        }
        std::cout << "视频属性: 宽度=" << width << ", 高度=" << height << ", FPS=" << fps
            << ", 总帧数=" << total_frames << std::endl;

        // 初始化输出视频（若 enable_log 启用）
        cv::VideoWriter out;
        bool enable_log = true; // 默认从 YAML 读取，此处硬编码为 true 以简化测试
        if (enable_log) {
            std::string save_path = "test_out_" + video_path.substr(video_path.find_last_of('/') + 1);
            std::string mjpg_path = save_path.substr(0, save_path.find_last_of('.')) + "_mjpg.avi";
            out.open(mjpg_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
            if (!out.isOpened()) {
                throw std::runtime_error("无法打开 VideoWriter: " + mjpg_path);
            }
            std::cout << "保存测试输出视频至: " << mjpg_path << std::endl;
        }

        // 逐帧处理
        int frame_count = 0;
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::cout << "视频读取结束，总计处理 " << frame_count << " 帧" << std::endl;
                break;
            }

            if (frame.empty()) {
                std::cerr << "帧 " << frame_count << ": 读取到空帧，跳过" << std::endl;
                continue;
            }

            if (frame.type() != CV_8UC3) {
                cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
                if (frame.type() != CV_8UC3) {
                    std::cerr << "帧 " << frame_count << ": 帧格式转换失败，类型: " << frame.type() << std::endl;
                    continue;
                }
            }

            // 调用 inference
            try {
                ActionInferenceResult result = pipeline.inference(frame);
                frame_count++;

                // 输出推理结果
                std::cout << "帧 " << frame_count << ": 检测到 " << result.online_targets.size() << " 个目标" << std::endl;
                for (size_t i = 0; i < result.labels.size(); ++i) {
                    std::cout << "  目标 " << i << ": 标签=" << result.labels[i]
                        << ", 概率=" << result.probs[i] << std::endl;
                }

                // 保存可视化帧
                if (enable_log && out.isOpened()) {
                    out.write(result.visualized_frame);
                }

                // 可选：显示帧（调试用，BM1684X 环境可能无显示器）
                // cv::imshow("Test Inference", result.visualized_frame);
                // if (cv::waitKey(1) == 27) break; // 按 ESC 退出
            }
            catch (const std::exception& e) {
                std::cerr << "帧 " << frame_count << ": 推理失败，错误: " << e.what() << std::endl;
                continue;
            }
        }

        // 释放资源
        cap.release();
        if (enable_log && out.isOpened()) {
            out.release();
            std::cout << "测试输出视频已保存" << std::endl;
        }

        // 可选：重置 pipeline 状态
        pipeline.reset();
        std::cout << "测试完成，pipeline 已重置" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "测试失败，错误: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <视频路径或图像路径> [设备ID]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    int dev_id = (argc > 2) ? std::stoi(argv[2]) : 0;

    // 运行测试函数
    test_inference("../models.yaml", input_path, dev_id);

    return 0;
}
