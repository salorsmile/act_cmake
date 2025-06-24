// falldetection_handle.cpp
#include "falldetection_handle.h"
#include "falldetection_pipeline.hpp"
#include <stdexcept>
#include <cstring>
#include <opencv2/opencv.hpp>

extern "C" {

    // 构造句柄
    EXPORT_API FalldetectionHandle falldetection_create(const char* config_path, int dev_id) {
        try {
            return new FalldetectionPipeline(config_path, dev_id);
        }
        catch (const std::exception& e) {
            std::cerr << "Error creating FalldetectionPipeline: " << e.what() << std::endl;
            return nullptr;
        }
    }

    // 销毁句柄
    EXPORT_API void falldetection_destroy(FalldetectionHandle handle) {
        if (handle) {
            delete static_cast<FalldetectionPipeline*>(handle);
        }
    }

    // 执行推理
    EXPORT_API int falldetection_inference(FalldetectionHandle handle,
        void* image,
        CActionInferenceResult* result) {
        if (!handle || !image || !result) {
            return -1; // 参数错误
        }

        try {
            // 将 void* 转换为 cv::Mat*
            cv::Mat* mat = static_cast<cv::Mat*>(image);
            if (mat->empty() || mat->type() != CV_8UC3) {
                std::cerr << "Invalid cv::Mat: empty or not CV_8UC3" << std::endl;
                return -1; // 无效的 cv::Mat
            }

            // 调用推理
            FalldetectionPipeline* pipeline = static_cast<FalldetectionPipeline*>(handle);
            ActionInferenceResult cpp_result = pipeline->inference(*mat);

            // 初始化 C 结果
            std::memset(result, 0, sizeof(CActionInferenceResult));

            // 验证字段一致性
            if (cpp_result.humans.size() != cpp_result.online_targets.targets.size() ||
                cpp_result.labels.size() != cpp_result.probs.size() ||
                cpp_result.labels.size() != cpp_result.humans.size()) {
                std::cerr << "Inconsistent result sizes: humans=" << cpp_result.humans.size()
                    << ", targets=" << cpp_result.online_targets.targets.size()
                    << ", labels=" << cpp_result.labels.size()
                    << ", probs=" << cpp_result.probs.size() << std::endl;
                return -3; // 数据不一致
            }

            // 转换 visualized_frame
            if (!cpp_result.visualized_frame.empty()) {
                result->frame_width = cpp_result.visualized_frame.cols;
                result->frame_height = cpp_result.visualized_frame.rows;
                result->frame_channels = cpp_result.visualized_frame.channels();
                size_t data_size = result->frame_width * result->frame_height * result->frame_channels;
                result->visualized_frame_data = (unsigned char*)malloc(data_size);
                if (!result->visualized_frame_data) {
                    std::cerr << "Failed to allocate visualized_frame_data" << std::endl;
                    return -2; // 内存分配失败
                }
                std::memcpy(result->visualized_frame_data, cpp_result.visualized_frame.data, data_size);
            }

            // 转换 humans
            result->human_count = cpp_result.humans.size();
            if (result->human_count > 0) {
                result->humans = (KeypointSet*)malloc(result->human_count * sizeof(KeypointSet));
                if (!result->humans) {
                    falldetection_free_result(result);
                    std::cerr << "Failed to allocate humans" << std::endl;
                    return -2;
                }
                for (int i = 0; i < result->human_count; ++i) {
                    result->humans[i].point_count = cpp_result.humans[i].size();
                    if (result->humans[i].point_count > 0) {
                        result->humans[i].points = (Point2f*)malloc(result->humans[i].point_count * sizeof(Point2f));
                        if (!result->humans[i].points) {
                            falldetection_free_result(result);
                            std::cerr << "Failed to allocate humans[" << i << "].points" << std::endl;
                            return -2;
                        }
                        for (int j = 0; j < result->humans[i].point_count; ++j) {
                            result->humans[i].points[j].x = cpp_result.humans[i][j].x;
                            result->humans[i].points[j].y = cpp_result.humans[i][j].y;
                        }
                    }
                    else {
                        result->humans[i].points = nullptr;
                    }
                }
            }

            // 转换 online_targets
            result->online_targets.target_count = cpp_result.online_targets.targets.size();
            if (result->online_targets.target_count > 0) {
                result->online_targets.targets = (CTrackEntry*)malloc(result->online_targets.target_count * sizeof(CTrackEntry));
                if (!result->online_targets.targets) {
                    falldetection_free_result(result);
                    std::cerr << "Failed to allocate online_targets.targets" << std::endl;
                    return -2;
                }
                for (int i = 0; i < result->online_targets.target_count; ++i) {
                    result->online_targets.targets[i].track_id = cpp_result.online_targets.targets[i].track_id;
                    result->online_targets.targets[i].state = cpp_result.online_targets.targets[i].state;
                    result->online_targets.targets[i].frame_id = cpp_result.online_targets.targets[i].frame_id;
                    result->online_targets.targets[i].tracklet_len = cpp_result.online_targets.targets[i].tracklet_len;
                    result->online_targets.targets[i].start_frame = cpp_result.online_targets.targets[i].start_frame;
                    result->online_targets.targets[i].score = cpp_result.online_targets.targets[i].score;
                    result->online_targets.targets[i].class_id = cpp_result.online_targets.targets[i].class_id;
                    result->online_targets.targets[i].tlbr = (float*)malloc(4 * sizeof(float));
                    if (!result->online_targets.targets[i].tlbr) {
                        falldetection_free_result(result);
                        std::cerr << "Failed to allocate tlbr for target " << i << std::endl;
                        return -2;
                    }
                    for (int j = 0; j < 4; ++j) {
                        result->online_targets.targets[i].tlbr[j] = cpp_result.online_targets.targets[i].tlbr[j];
                    }
                }
            }

            // 转换 labels 和 probs
            result->label_count = cpp_result.labels.size();
            if (result->label_count > 0) {
                result->labels = (char**)malloc(result->label_count * sizeof(char*));
                result->probs = (float*)malloc(result->label_count * sizeof(float));
                if (!result->labels || !result->probs) {
                    falldetection_free_result(result);
                    std::cerr << "Failed to allocate labels or probs" << std::endl;
                    return -2;
                }
                for (int i = 0; i < result->label_count; ++i) {
                    result->probs[i] = cpp_result.probs[i];
                    size_t len = cpp_result.labels[i].length() + 1;
                    result->labels[i] = (char*)malloc(len);
                    if (!result->labels[i]) {
                        falldetection_free_result(result);
                        std::cerr << "Failed to allocate labels[" << i << "]" << std::endl;
                        return -2;
                    }
                    std::strcpy(result->labels[i], cpp_result.labels[i].c_str());
                }
            }

            return 0; // 成功
        }
        catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            falldetection_free_result(result);
            return -3; // 推理异常
        }
    }

    // 释放推理结果的内存
    EXPORT_API void falldetection_free_result(CActionInferenceResult* result) {
        if (!result) return;

        // 释放 visualized_frame_data
        if (result->visualized_frame_data) {
            free(result->visualized_frame_data);
            result->visualized_frame_data = nullptr;
        }

        // 释放 humans
        if (result->humans) {
            for (int i = 0; i < result->human_count; ++i) {
                if (result->humans[i].points) {
                    free(result->humans[i].points);
                }
            }
            free(result->humans);
            result->humans = nullptr;
            result->human_count = 0;
        }

        // 释放 online_targets
        if (result->online_targets.targets) {
            for (int i = 0; i < result->online_targets.target_count; ++i) {
                if (result->online_targets.targets[i].tlbr) {
                    free(result->online_targets.targets[i].tlbr);
                }
            }
            free(result->online_targets.targets);
            result->online_targets.targets = nullptr;
            result->online_targets.target_count = 0;
        }

        // 释放 labels 和 probs
        if (result->labels) {
            for (int i = 0; i < result->label_count; ++i) {
                if (result->labels[i]) {
                    free(result->labels[i]);
                }
            }
            free(result->labels);
            result->labels = nullptr;
        }
        if (result->probs) {
            free(result->probs);
            result->probs = nullptr;
        }
        result->label_count = 0;
    }

    // 重置状态
    EXPORT_API void falldetection_reset(FalldetectionHandle handle) {
        if (handle) {
            try {
                static_cast<FalldetectionPipeline*>(handle)->reset();
            }
            catch (const std::exception& e) {
                std::cerr << "Error during reset: " << e.what() << std::endl;
            }
        }
    }

} // extern "C"