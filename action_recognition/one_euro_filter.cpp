
#include "one_euro_filter.hpp"
#include <cmath>

#ifndef CV_PI
#define CV_PI 3.14159265358979323846
#endif

OneEuroFilter::OneEuroFilter(float te, float mincutoff, float beta, float dcutoff)
	: te_(te), mincutoff_(mincutoff), beta_(beta), dcutoff_(dcutoff) {
	alpha_ = alpha(mincutoff_);
	dalpha_ = alpha(dcutoff_);
}

OneEuroFilter::OneEuroFilter(float te)
	: te_(te) {

	mincutoff_ = 1.0f;
	beta_ = 0.007f;
	dcutoff_ = 1.0f;
	alpha_ = alpha(mincutoff_);
	dalpha_ = alpha(dcutoff_);
}

std::vector<cv::Point2f> OneEuroFilter::predict(const std::vector<cv::Point2f>& x, float te) {
	std::vector<cv::Point2f> result = x;
	if (x_prev_.empty()) {
		x_prev_ = x;
		dx_prev_.resize(x.size(), cv::Point2f(0, 0));
		return result;
	}

	for (size_t i = 0; i < x.size(); ++i) {
		cv::Point2f edx = (x[i] - x_prev_[i]) / te;
		dx_prev_[i] = dx_prev_[i] + dalpha_ * (edx - dx_prev_[i]);
		float cutoff = mincutoff_ + beta_ * cv::norm(dx_prev_[i]);
		alpha_ = alpha(cutoff);
		result[i] = x_prev_[i] + alpha_ * (x[i] - x_prev_[i]);
	}

	x_prev_ = result;
	return result;
}

float OneEuroFilter::alpha(float cutoff) {
	float tau = 1.0f / (2 * CV_PI * cutoff);
	return 1.0f / (1.0f + tau / te_);
}
