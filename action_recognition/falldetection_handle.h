// falldetection_handle.h
#ifndef FALLDETECTION_HANDLE_H
#define FALLDETECTION_HANDLE_H

#ifdef __cplusplus
extern "C" {
#endif

    // 定义导出宏
#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// 不透明句柄类型
    typedef void* FalldetectionHandle;

    // 表示单个跟踪目标的信息（C 版本）
    typedef struct {
        int track_id;         // 跟踪ID
        int state;            // 跟踪状态
        float* tlbr;          // 边界框 (top-left-bottom-right)，长度为4
        int frame_id;         // 当前帧ID
        int tracklet_len;     // 跟踪持续时间
        int start_frame;      // 跟踪开始帧
        float score;          // 检测得分
        int class_id;         // 类别ID
    } CTrackEntry;

    // 表示跟踪信息，包含多个跟踪目标（C 版本）
    typedef struct {
        CTrackEntry* targets;  // 跟踪目标数组
        int target_count;      // 目标数量
    } CTrackInfo;

    // 表示关键点信息
    typedef struct {
        float x;
        float y;
    } Point2f;

    // 表示一个人的关键点集合
    typedef struct {
        Point2f* points;      // 关键点数组
        int point_count;      // 关键点数量
    } KeypointSet;

    // 表示推理结果（C 版本）
    typedef struct {
        unsigned char* visualized_frame_data; // 可视化帧的图像数据（BGR格式）
        int frame_width;                      // 帧宽度
        int frame_height;                     // 帧高度
        int frame_channels;                   // 帧通道数
        KeypointSet* humans;                  // 人的关键点数组
        int human_count;                      // 人数
        CTrackInfo online_targets;            // 跟踪信息
        char** labels;                        // 动作标签数组
        float* probs;                         // 动作概率数组
        int label_count;                      // 标签数量
    } CActionInferenceResult;

    // 构造句柄
    EXPORT_API FalldetectionHandle falldetection_create(const char* config_path, int dev_id);

    // 销毁句柄
    EXPORT_API void falldetection_destroy(FalldetectionHandle handle);

    // 执行推理（接受 cv::Mat 作为 void*）
    EXPORT_API int falldetection_inference(FalldetectionHandle handle,
        void* image,
        CActionInferenceResult* result);

    // 释放推理结果的内存
    EXPORT_API void falldetection_free_result(CActionInferenceResult* result);

    // 重置状态
    EXPORT_API void falldetection_reset(FalldetectionHandle handle);

#ifdef __cplusplus
}
#endif

#endif // FALLDETECTION_HANDLE_H