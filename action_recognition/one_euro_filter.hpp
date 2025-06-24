#pragma once

#ifndef ONE_EURO_FILTER_HPP
#define ONE_EURO_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class OneEuroFilter {
public:
	// ���캯������ʼ���˲�����
	OneEuroFilter(float te);
	OneEuroFilter(float te, float mincutoff, float beta, float dcutoff);
	std::vector<cv::Point2f> predict(const std::vector<cv::Point2f>& x, float te);
	float alpha(float cutoff);

	std::vector<cv::Point2f> x_prev_;  // ǰһ֡�Ĺؼ���
	std::vector<cv::Point2f> dx_prev_; // ǰһ֡�ĵ���
	float te_; // ʱ�䲽��
	float mincutoff_; // ��С��ֹƵ��
	float beta_; // �ٶ�Ȩ��
	float dcutoff_; // ������ֹƵ��
	float alpha_; // ��ǰ alpha ֵ
	float dalpha_; // ���� alpha ֵ
};

#endif // ONE_EURO_FILTER_HPP