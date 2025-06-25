#include "falldetection_pipeline.hpp"
#include <stdexcept>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>

FalldetectionPipeline::FalldetectionPipeline(const std::string& config_path, int dev_id)
	: dev_id_(dev_id), counter_(0), text_duration_(0) {
	parse_config(config_path);
	init_models();
}

FalldetectionPipeline::~FalldetectionPipeline() {
	// 释放所有资源
	reset(); // 清空状态变量
	yolov5_.reset();
	bytetrack_.reset();
	hrnet_pose_.reset();
	classifier_.reset();
	filter_.reset();
	scaled_filter_.reset();
	time_stamp_.reset();
	handle_.reset();
}

void FalldetectionPipeline::parse_config(const std::string& config_path) {
	// 设置默认值
	args_.img_shape = { 256, 192 };
	args_.device = "tpu";
	args_.estimator_bmodel_path = "models/pose_estimator_int8.bmodel";
	args_.detector_bmodel_path = "models/detector_int8_4b.bmodel";
	args_.classifier_bmodel_path = "models/action_recognition_fp32_1b.bmodel";
	args_.class_names = { "fall", "normal" };
	args_.seg = 30;
	args_.num_joint = 17;
	args_.num_classes = 2;
	args_.channels = 2;
	args_.detector_prob_threshold = 0.7f;
	args_.disable_filter = false;
	args_.skeleton_visible = true;
	args_.enable_log = true;
	args_.cfg_path = config_path;
	args_.save_result = true;
	args_.visualized_frame = true;

	// 读取 YAML 文件
	try {
		YAML::Node config = YAML::LoadFile(args_.cfg_path);
		if (config["models"] && config["models"]["fall_recognition"]) {
			const auto& fall_recog = config["models"]["fall_recognition"];

			// 读取模型路径
			if (fall_recog["estimator_bmodel_path"]) {
				args_.estimator_bmodel_path = fall_recog["estimator_bmodel_path"].as<std::string>();
			}
			if (fall_recog["detector_bmodel_path"]) {
				args_.detector_bmodel_path = fall_recog["detector_bmodel_path"].as<std::string>();
			}
			if (fall_recog["classifier_bmodel_path"]) {
				args_.classifier_bmodel_path = fall_recog["classifier_bmodel_path"].as<std::string>();
			}

			// 读取图像形状
			if (fall_recog["img_shape"]) {
				args_.img_shape = fall_recog["img_shape"].as<std::vector<int>>();
			}

			// 读取检测阈值
			if (fall_recog["detector_prob_threshold"]) {
				args_.detector_prob_threshold = fall_recog["detector_prob_threshold"].as<float>();
			}

			// 读取滤波和骨骼可视化设置
			if (fall_recog["disable_filter"]) {
				args_.disable_filter = fall_recog["disable_filter"].as<bool>();
			}
			if (fall_recog["skeleton_visible"]) {
				args_.skeleton_visible = fall_recog["skeleton_visible"].as<bool>();
			}
			if (fall_recog["visualized_frame"]) {
				args_.visualized_frame = fall_recog["visualized_frame"].as<bool>();
			}

			// 读取类名
			if (fall_recog["class_names"]) {
				args_.class_names = fall_recog["class_names"].as<std::vector<std::string>>();
				args_.num_classes = args_.class_names.size();
			}

			// 读取日志设置
			if (fall_recog["enable_log"]) {
				args_.enable_log = fall_recog["enable_log"].as<bool>();
			}
		}
	}
	catch (const YAML::Exception& e) {
		std::cerr << "解析 YAML 文件失败: " << e.what() << std::endl;
		std::cerr << "使用默认参数继续初始化" << std::endl;
	}
}

void FalldetectionPipeline::init_models() {
	handle_ = std::make_shared<BMNNHandle>(dev_id_);
	bm_handle_t h = handle_->handle();

	auto ts = std::make_shared<TimeStamp>();
	auto bm_ctx_detector = std::make_shared<BMNNContext>(handle_, args_.detector_bmodel_path.c_str());
	yolov5_ = std::make_unique<YoloV5>(bm_ctx_detector);
	yolov5_->Init(args_.detector_prob_threshold, 0.6f, "");
	yolov5_->enableProfile(ts);
	time_stamp_ = ts;

	bytetrack_params params;
	params.track_thresh = 0.1f;
	params.track_buffer = 30;
	params.match_thresh = 0.80f;
	bytetrack_ = std::make_unique<BYTETracker>(params);

	auto bm_ctx_pose = std::make_shared<BMNNContext>(handle_, args_.estimator_bmodel_path.c_str());
	hrnet_pose_ = std::make_unique<HRNetPose>(bm_ctx_pose);
	hrnet_pose_->Init(false, "");
	hrnet_pose_->enableProfile(ts);

	classifier_ = std::make_unique<ActionRecognition>(args_.classifier_bmodel_path,
		args_.seg, args_.num_joint,
		args_.num_classes, args_.channels,
		dev_id_);

	filter_ = std::make_unique<OneEuroFilter>(1.0f / 30.0f, 1.0f, 0.007f, 1.0f);
	scaled_filter_ = std::make_unique<OneEuroFilter>(1.0f / 30.0f, 1.0f, 0.007f, 1.0f);
}

void FalldetectionPipeline::video_inference() {
	cv::VideoCapture cap(args_.video_path);
	if (!cap.isOpened()) {
		throw std::runtime_error("无法打开视频文件: " + args_.video_path);
	}

	int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	if (width <= 0 || height <= 0 || fps <= 0) {
		throw std::runtime_error("无效的视频属性: width=" + std::to_string(width) +
			", height=" + std::to_string(height) +
			", fps=" + std::to_string(fps));
	}

	cv::VideoWriter out;
	if (args_.save_result && args_.enable_log) {
		std::string save_path = "out_" + args_.video_path.substr(args_.video_path.find_last_of('/') + 1);
		std::string mjpg_path = save_path.substr(0, save_path.find_last_of('.')) + "_mjpg.avi";
		out.open(mjpg_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
		if (!out.isOpened()) {
			throw std::runtime_error("无法打开 VideoWriter: " + mjpg_path);
		}
		std::cout << "保存视频至: " << mjpg_path << std::endl;
	}

	while (true) {
		cv::Mat frame;
		if (!cap.read(frame)) {
			break;
		}

		if (frame.empty()) {
			std::cerr << "读取到空帧，跳过" << std::endl;
			continue;
		}

		if (frame.type() != CV_8UC3) {
			cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
			if (frame.type() != CV_8UC3) {
				std::cerr << "帧格式转换失败，类型: " << frame.type() << std::endl;
				continue;
			}
		}

		ActionInferenceResult result = inference(frame);
		if (args_.save_result && args_.enable_log) {
			out.write(result.visualized_frame);
		}
	}

	if (args_.save_result && args_.enable_log) {
		out.release();
	}
	cap.release();
}



ActionInferenceResult FalldetectionPipeline::inference(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::runtime_error("输入帧为空");
    }

    cv::Mat frame_copy = frame.clone();

    //cv::imwrite("./frame_copy.jpg", frame_copy);

    if (frame_copy.type() != CV_8UC3) {
        cv::cvtColor(frame_copy, frame_copy, cv::COLOR_YUV2BGR);
        if (frame_copy.type() != CV_8UC3) {
            throw std::runtime_error("帧格式转换失败，类型: " + std::to_string(frame_copy.type()));
        }
    }

    double before = cv::getTickCount() / cv::getTickFrequency() * 1000;

    bm_handle_t handle;
    bm_status_t status = bm_dev_request(&handle, dev_id_);
    if (status != BM_SUCCESS) {
        throw std::runtime_error("无法请求设备，状态=" + std::to_string(status));
    }

    bm_image bm_img;
    bm_image_create(handle, frame_copy.rows, frame_copy.cols, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &bm_img);
    bm_image_alloc_dev_mem(bm_img, BMCV_IMAGE_FOR_IN);
    cv::bmcv::toBMI(frame_copy, &bm_img);

    double det_time = cv::getTickCount() / cv::getTickFrequency() * 1000;

    std::vector<YoloV5BoxVec> yolov5_boxes;
    std::vector<bm_image> batch_imgs = { bm_img };
    if (!yolov5_) {
        bm_image_destroy(bm_img);
        bm_dev_free(handle);
        throw std::runtime_error("YoloV5 未初始化");
    }
    yolov5_->Detect(batch_imgs, yolov5_boxes);

    double track_time = cv::getTickCount() / cv::getTickFrequency() * 1000;

    // 清空当前帧的推理状态
    humans_.clear();
    scaled_humans_.clear();
    labels_.clear();
    probs_.clear();
    online_targets_.targets.clear();

    humans_.reserve(10);
    scaled_humans_.reserve(10);
    labels_.reserve(10);
    probs_.reserve(10);

    if (!yolov5_boxes.empty() && !yolov5_boxes[0].empty()) {
        STracks stracks; // 临时存储 BYTETracker 的输出
        bytetrack_->update(stracks, yolov5_boxes[0]);
        counter_++;

        // 将 STracks 转换为 TrackInfo
        online_targets_.targets.reserve(stracks.size());
        for (const auto& box : stracks) {
            TrackEntry entry;
            entry.track_id = box->track_id;
            entry.state = box->state;
            entry.tlbr = box->tlbr; // 直接使用 tlbr
            entry.frame_id = box->frame_id;
            entry.tracklet_len = box->tracklet_len;
            entry.start_frame = box->start_frame;
            entry.score = box->score;
            entry.class_id = box->class_id;
            online_targets_.targets.push_back(entry);
        }

        for (const auto& box : online_targets_.targets) {
            YoloV5Box person_box;
            // 从 tlbr 转换为 tlwh
            person_box.x = box.tlbr[0]; // top-left x
            person_box.y = box.tlbr[1]; // top-left y
            person_box.width = box.tlbr[2] - box.tlbr[0]; // right - left
            person_box.height = box.tlbr[3] - box.tlbr[1]; // bottom - top
            person_box.score = box.score;
            person_box.class_id = box.class_id;

            std::vector<cv::Point2f> keypoints;
            std::vector<float> maxvals;
            std::vector<cv::Mat> heatmaps;
            hrnet_pose_->poseEstimate(bm_img, person_box, keypoints, maxvals, heatmaps);

            if (!args_.disable_filter && !keypoints.empty()) {
                keypoints = filter_->predict(keypoints, 1.0f / 30.0f);
                std::vector<cv::Point2f> scaled_keypoints = keypoints;
                for (auto& pt : scaled_keypoints) {
                    pt.x /= 384.0f;
                    pt.y /= 512.0f;
                }
                scaled_keypoints = scaled_filter_->predict(scaled_keypoints, 1.0f / 30.0f);
                humans_.push_back(keypoints);
                scaled_humans_.push_back(scaled_keypoints);
            }
            else {
                humans_.push_back(keypoints);
                std::vector<cv::Point2f> scaled_keypoints = keypoints;
                for (auto& pt : scaled_keypoints) {
                    pt.x /= 384.0f;
                    pt.y /= 512.0f;
                }
                scaled_humans_.push_back(scaled_keypoints);
            }
        }

        double estimation_time = cv::getTickCount() / cv::getTickFrequency() * 1000;

        // 清理过期的跟踪目标
        std::vector<int> active_track_ids;
        active_track_ids.reserve(online_targets_.targets.size());
        for (const auto& box : online_targets_.targets) {
            active_track_ids.push_back(box.track_id);
        }
        for (auto it = frames_buffer_.begin(); it != frames_buffer_.end();) {
            if (std::find(active_track_ids.begin(), active_track_ids.end(), it->first) == active_track_ids.end()) {
                it = frames_buffer_.erase(it);
            }
            else {
                ++it;
            }
        }

        // 更新 frames_buffer_ 和动作识别
        for (size_t idx = 0; idx < online_targets_.targets.size(); ++idx) {
            int track_id = online_targets_.targets[idx].track_id;
            if (args_.enable_log) {
                std::cout << "frame " << counter_ << ": targets " << idx << ", track_id=" << track_id << "\n";
            }
            frames_buffer_[track_id].push_back(scaled_humans_[idx]);
            if (frames_buffer_[track_id].size() > static_cast<size_t>(args_.seg)) {
                frames_buffer_[track_id].erase(frames_buffer_[track_id].begin());
            }
            if (frames_buffer_[track_id].size() >= static_cast<size_t>(args_.seg)) {
                auto [label, prob] = classifier_->infer(frames_buffer_[track_id]);
                if (label == args_.class_names[0]) { // "fall"
                    text_duration_ = 30;
                }
                labels_.push_back(label);
                probs_.push_back(prob);
            }
            else {
                labels_.push_back("Tracking");
                probs_.push_back(0.0f);
            }
        }

        double end = cv::getTickCount() / cv::getTickFrequency() * 1000;

        if (args_.enable_log) {
            std::cout << "frame " << counter_ << ": detected " << online_targets_.targets.size() << " targets\n";
            std::cout << "activate track ID: ";
            for (const auto& box : online_targets_.targets) {
                std::cout << box.track_id << " ";
            }
            std::cout << "\nframes_buffer_size: " << frames_buffer_.size() << "\n";
            for (size_t i = 0; i < online_targets_.targets.size(); ++i) {
                std::cout << "  target " << online_targets_.targets[i].track_id
                    << ": label=" << labels_[i] << ", prob=" << probs_[i]
                    << ", kpts=" << humans_[i].size() << "\n";
            }
            std::cout << "总耗时: " << (end - before) << "ms/帧"
                << "\t检测: " << (det_time - before) << "ms"
                << "\t跟踪: " << (track_time - det_time) << "ms"
                << "\t姿态估计: " << (estimation_time - track_time) << "ms"
                << "\t动作识别: " << (end - estimation_time) << "ms\n";
        }
    }

    if (text_duration_ > 0) {
        cv::putText(frame_copy, "Fall detected " + std::to_string(30 - text_duration_) + " frame ago!!",
            cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 1);
        text_duration_--;
    }

    // 准备返回结果
    ActionInferenceResult result;

    if (args_.visualized_frame) {
        result.visualized_frame = visualize(frame_copy, humans_, online_targets_, labels_, probs_, args_.skeleton_visible);

    }
    else {
        result.visualized_frame = cv::Mat();
    }

    result.humans = humans_;
    result.online_targets = online_targets_;
    result.labels = labels_;
    result.probs = probs_;

    bm_image_destroy(bm_img);
    bm_dev_free(handle);

    return result;
}

void FalldetectionPipeline::reset() {
    counter_ = 0;
    text_duration_ = 0;
    frames_buffer_.clear();
    humans_.clear();
    scaled_humans_.clear();
    online_targets_.targets.clear();
    labels_.clear();
    probs_.clear();
    bytetrack_ = std::make_unique<BYTETracker>(bytetrack_params{ 0.1f, 30, 0.95f });
}

cv::Mat FalldetectionPipeline::visualize(cv::Mat frame, const std::vector<std::vector<cv::Point2f>>& keypoints,
    const TrackInfo& boxes, const std::vector<std::string>& labels,
    const std::vector<float>& probs, bool vis_skeleton) {
    for (size_t i = 0; i < boxes.targets.size(); ++i) {
        const auto& box = boxes.targets[i];
     
        float x1 = box.tlbr[0];
        float y1 = box.tlbr[1];
        float x2 = box.tlbr[2];
        float y2 = box.tlbr[3];
        cv::rectangle(frame, cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(0, 0, 255), 2);

        std::string label_text = probs[i] == 0 ? labels[i] : labels[i] + " : " + std::to_string(probs[i] * 100) + "%";
        cv::putText(frame, label_text, cv::Point(x1, y1 + 20),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        if (vis_skeleton && i < keypoints.size()) {
            hrnet_pose_->drawPose(keypoints[i], frame);
        }
    }

    //cv::imwrite("./frame_vis.jpg", frame);
    return frame;

}
