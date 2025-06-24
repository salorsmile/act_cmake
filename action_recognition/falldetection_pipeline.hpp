
#ifndef FALLDETECTION_API_HPP
#define FALLDETECTION_API_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "bmnn_utils.h"
#include "bm_wrapper.hpp"
#include "yolov5.hpp"
#include "hrnet_pose.hpp"
#include "bytetrack.h"
#include "one_euro_filter.hpp"
#include "utils.hpp"
#include "action_recognition.hpp"


// 定义导出宏
#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// 前向声明
class BMNNHandle;
class YoloV5;
class BYTETracker;
class HRNetPose;
class ActionRecognition;
class OneEuroFilter;
class TimeStamp;


// 定义单个跟踪目标的信息
struct TrackEntry {
	int track_id;          // 跟踪ID
	int state;             // 跟踪状态
	std::vector<float> tlbr; // 边界框 (top-left-bottom-right)
	int frame_id;          // 当前帧ID
	int tracklet_len;      // 跟踪持续时间
	int start_frame;       // 跟踪开始帧
	float score;           // 检测得分
	int class_id;          // 类别ID
};

// 定义跟踪信息，包含多个跟踪目标
struct TrackInfo {
	std::vector<TrackEntry> targets; // 跟踪目标列表
};

struct EXPORT_API ActionInferenceResult {
	cv::Mat visualized_frame; // 可视化后的帧
	std::vector<std::vector<cv::Point2f>> humans; // 人的关键点
	TrackInfo online_targets; // 跟踪信息
	std::vector<std::string> labels; // 动作标签
	std::vector<float> probs; // 动作概率
};

class EXPORT_API FalldetectionPipeline {
public:
	FalldetectionPipeline(const std::string& config_path, int dev_id);

	// 析构函数
	~FalldetectionPipeline();

	// 处理视频流
	void video_inference();// 测试函数

	// 处理单帧图像
	ActionInferenceResult inference(const cv::Mat& frame);

	// 重置状态
	void reset();

private:
	struct Args {
		std::vector<int> img_shape;
		std::string device;
		std::string estimator_bmodel_path;
		std::string detector_bmodel_path;
		std::string classifier_bmodel_path;
		std::vector<std::string> class_names;
		int seg;
		int num_joint;
		int num_classes;
		int channels;
		float detector_prob_threshold;
		bool disable_filter;
		bool skeleton_visible;
		bool enable_log;
		std::string video_path;
		std::string cfg_path;
		bool save_result;
		bool visualized_frame;
	};

	void parse_config(const std::string& config_path);
	void init_models();
	cv::Mat visualize(cv::Mat frame, const std::vector<std::vector<cv::Point2f>>& keypoints,
		const TrackInfo& boxes, const std::vector<std::string>& labels,
		const std::vector<float>& probs, bool vis_skeleton);


	Args args_;
	int dev_id_;
	std::shared_ptr<BMNNHandle> handle_;
	std::unique_ptr<YoloV5> yolov5_;
	std::unique_ptr<BYTETracker> bytetrack_;
	std::unique_ptr<HRNetPose> hrnet_pose_;
	std::unique_ptr<ActionRecognition> classifier_;
	std::unique_ptr<OneEuroFilter> filter_;
	std::unique_ptr<OneEuroFilter> scaled_filter_;
	std::shared_ptr<TimeStamp> time_stamp_;
	// 状态变量
	int counter_;
	int text_duration_;
	std::map<int, std::vector<std::vector<cv::Point2f>>> frames_buffer_;
	// 推理状态变量
	std::vector<std::vector<cv::Point2f>> humans_;
	std::vector<std::vector<cv::Point2f>> scaled_humans_;
	TrackInfo online_targets_;
	std::vector<std::string> labels_;
	std::vector<float> probs_;
};

#endif // FALLDETECTION_API_HPP