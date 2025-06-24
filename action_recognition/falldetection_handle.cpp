// falldetection_handle.cpp
#include "falldetection_handle.h"
#include "falldetection_pipeline.hpp"
#include <stdexcept>
#include <cstring>

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
        const unsigned char* frame_data,
        int width,
        int height,
        int channels,
        CActionInferenceResult* result) {
        if (!handle || !frame_data || !result || width <= 0 || height <= 0 || channels != 3) {
            return -1; // 参数错误
        }

        try {
            // 转换为 cv::Mat
            cv::Mat frame(height, width, CV_8UC3);
            std::memcpy(frame.data, frame_data, width * height * channels);

            // 调用推理
            FalldetectionPipeline* pipeline = static_cast<FalldetectionPipeline*>(handle);
            ActionInferenceResult cpp_result = pipeline->inference(frame);

            // 初始化 C 结果
            std::memset(result, 0, sizeof(CActionInferenceResult));

            // 转换 visualized_frame
            if (!cpp_result.visualized_frame.empty()) {
                result->frame_width = cpp_result.visualized_frame.cols;
                result->frame_height = cpp_result.visualized_frame.rows;
                result->frame_channels = cpp_result.visualized_frame.channels();
                size_t data_size = result->frame_width * result->frame_height * result->frame_channels;
                result->visualized_frame_data = (unsigned char*)malloc(data_size);
                if (!result->visualized_frame_data) {
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
                    return -2;
                }
                for (int i = 0; i < result->human_count; ++i) {
                    result->humans[i].point_count = cpp_result.humans[i].size();
                    if (result->humans[i].point_count > 0) {
                        result->humans[i].points = (Point2f*)malloc(result->humans[i].point_count * sizeof(Point2f));
                        if (!result->humans[i].points) {
                            falldetection_free_result(result);
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
                result->online_targets.targets = (TrackEntry*)malloc(result->online_targets.target_count * sizeof(TrackEntry));
                if (!result->online_targets.targets) {
                    falldetection_free_result(result);
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
                    return -2;
                }
                for (int i = 0; i < result->label_count; ++i) {
                    result->probs[i] = cpp_result.probs[i];
                    size_t len = cpp_result.labels[i].length() + 1;
                    result->labels[i] = (char*)malloc(len);
                    if (!result->labels[i]) {
                        falldetection_free_result(result);
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
            free(result->online_targets.targetsMotor);
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