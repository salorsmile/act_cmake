#pragma once

#ifndef ONE_EURO_FILTER_HPP
#define ONE_EURO_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class OneEuroFilter {
public:
	// 构造函数，初始化滤波参数
	OneEuroFilter(float te);
	OneEuroFilter(float te, float mincutoff, float beta, float dcutoff);
	std::vector<cv::Point2f> predict(const std::vector<cv::Point2f>& x, float te);
	float alpha(float cutoff);

	std::vector<cv::Point2f> x_prev_;  // 前一帧的关键点
	std::vector<cv::Point2f> dx_prev_; // 前一帧的导数
	float te_; // 时间步长
	float mincutoff_; // 最小截止频率
	float beta_; // 速度权重
	float dcutoff_; // 导数截止频率
	float alpha_; // 当前 alpha 值
	float dalpha_; // 导数 alpha 值
};

#endif // ONE_EURO_FILTER_HPP