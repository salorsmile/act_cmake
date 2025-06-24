
#ifndef ACTION_RECOGNITION_HPP
#define ACTION_RECOGNITION_HPP

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bmnn_utils.h"
#include "bm_wrapper.hpp"
#include "bmlib_runtime.h"

class ActionRecognition {
public:
    // 构造函数
    ActionRecognition(const std::string& model_path, int seg, int num_joint,
        int num_classes, int channels, int dev_id);

    // 析构函数
    ~ActionRecognition();

    // 推理方法，基于关键点序列预测动作
    std::pair<std::string, float> infer(const std::vector<std::vector<cv::Point2f>>& frames_buffer);

private:
    std::shared_ptr<BMNNContext> bm_ctx_;
    std::shared_ptr<BMNNNetwork> network_;
    bm_shape_t input_shape_;
    bm_shape_t output_shape_;
    int seg_;
    int num_joint_;
    int num_classes_;
    int channels_;
    std::vector<std::string> labels_; // 动作标签列表
};

#endif // ACTION_RECOGNITION_HPP
