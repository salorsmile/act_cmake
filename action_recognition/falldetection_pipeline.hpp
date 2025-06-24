
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
#ifdef __linux__
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT_API
#endif

// 前向声明
class BMNNHandle;
class YoloV5;
class BYTETracker;
class HRNetPose;
class ActionRecognition;
class OneEuroFilter;
class TimeStamp;
using STracks = std::vector<std::shared_ptr<STrack>>;

struct EXPORT_API ActionInferenceResult {
	cv::Mat visualized_frame; // 可视化后的帧
	std::vector<std::vector<cv::Point2f>> humans; // 人的关键点
	STracks online_targets; // 跟踪信息
	std::vector<std::string> labels; // 动作标签
	std::vector<float> probs; // 动作概率
};

class EXPORT_API FalldetectionPipeline {
public:
	FalldetectionPipeline(const std::string& config_path, int dev_id);

	// 析构函数
	~FalldetectionPipeline();

	// 处理视频流
	void video_inference();

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
		const STracks& boxes, const std::vector<std::string>& labels,
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
	STracks online_targets_;
	std::vector<std::string> labels_;
	std::vector<float> probs_;
};

#endif // FALLDETECTION_API_HPP