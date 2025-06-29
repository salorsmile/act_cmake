//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "bytetrack.h"

#include <fstream>

BYTETracker::BYTETracker(const bytetrack_params& params) {
	this->track_thresh = params.track_thresh;
	this->match_thresh = params.match_thresh;
	this->frame_rate = params.frame_rate;
	this->track_buffer = params.track_buffer;
	this->min_box_area = params.min_box_area;
	this->frame_id = 0;
	this->max_time_lost = int(this->frame_rate / 30.0 * this->track_buffer);
	this->kalman_filter = std::make_shared<KalmanFilter>();
	std::cout << "Init ByteTrack!" << std::endl;
}

void BYTETracker::enableProfile(TimeStamp* ts) { m_ts = ts; }

BYTETracker::~BYTETracker() {}

void BYTETracker::update(STracks& output_stracks,
	const std::vector<YoloV5Box>& objects) {
	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	STracks activated_stracks;
	STracks refind_stracks;
	STracks detections;
	STracks detections_low;
	STracks detections_cp;
	STracks tracked_stracks_swap;
	STracks resa, resb;
	STracks temp_tracked_stracks;
	STracks temp_lost_stracks;
	STracks temp_removed_stracks;
	STracks unconfirmed;
	STracks strack_pool;
	STracks r_tracked_stracks;

	if (objects.size() > 0) {
		for (int i = 0; i < objects.size(); i++) {
			std::vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].x;
			tlbr_[1] = objects[i].y;
			tlbr_[2] = objects[i].x + objects[i].width;
			tlbr_[3] = objects[i].y + objects[i].height;

			float score = objects[i].score;
			int class_id = objects[i].class_id;

			std::shared_ptr<STrack> strack = std::make_shared<STrack>(
				STrack::tlbr_to_tlwh(tlbr_), score, class_id);
			if (score >= track_thresh) {
				detections.push_back(strack);
			}
			else {
				detections_low.push_back(strack);
			}
		}
	}
	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (!this->tracked_stracks[i]->is_activated)
			unconfirmed.push_back(this->tracked_stracks[i]);
		else
			temp_tracked_stracks.push_back(this->tracked_stracks[i]);
	}
	////////////////// Step 2: First association, with IoU //////////////////
	joint_stracks(temp_tracked_stracks, this->lost_stracks, strack_pool);
	STrack::multi_predict(strack_pool, this->kalman_filter);

	std::vector<std::vector<float>> dists;
	int dist_size = strack_pool.size(), dist_size_size = detections.size();
	iou_distance(strack_pool, detections, dists);

	std::vector<std::vector<int>> matches;
	std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches,
		u_track, u_detection);
	for (int i = 0; i < matches.size(); i++) {
		std::shared_ptr<STrack> track = strack_pool[matches[i][0]];
		std::shared_ptr<STrack> det = detections[matches[i][1]];
		if (track->state == TrackState::Tracked) {
			track->update(this->kalman_filter, det, this->frame_id);
			activated_stracks.push_back(track);
		}
		else {
			track->re_activate(this->kalman_filter, det, this->frame_id, false);
			refind_stracks.push_back(track);
		}
	}
	////////////////// Step 3: Second association, using low score dets
	/////////////////////
	for (int i = 0; i < u_detection.size(); i++) {
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	for (int i = 0; i < u_track.size(); i++) {
		if (strack_pool[u_track[i]]->state == TrackState::Tracked) {
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	iou_distance(r_tracked_stracks, detections, dists);
	dist_size = r_tracked_stracks.size();
	dist_size_size = detections.size();

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track,
		u_detection);

	for (int i = 0; i < matches.size(); i++) {
		std::shared_ptr<STrack> track = r_tracked_stracks[matches[i][0]];
		std::shared_ptr<STrack> det = detections[matches[i][1]];
		if (track->state == TrackState::Tracked) {
			track->update(this->kalman_filter, det, this->frame_id);
			activated_stracks.push_back(track);
		}
		else {
			track->re_activate(this->kalman_filter, det, this->frame_id, false);
			refind_stracks.push_back(track);
		}
	}

	for (int i = 0; i < u_track.size(); i++) {
		std::shared_ptr<STrack> track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost) {
			track->mark_lost();
			temp_lost_stracks.push_back(track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	iou_distance(unconfirmed, detections, dists);
	dist_size = unconfirmed.size();
	dist_size_size = detections.size();

	matches.clear();
	std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches,
		u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++) {
		unconfirmed[matches[i][0]]->update(
			this->kalman_filter, detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++) {
		std::shared_ptr<STrack> track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		temp_removed_stracks.push_back(track);
	}
	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < u_detection.size(); i++) {
		std::shared_ptr<STrack> track = detections[u_detection[i]];
		if (track->score < this->track_thresh) continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(track);
	}
	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++) {
		if (this->frame_id - this->lost_stracks[i]->end_frame() >
			this->max_time_lost) {
			this->lost_stracks[i]->mark_removed();
			temp_removed_stracks.push_back(this->lost_stracks[i]);
		}
	}

	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (this->tracked_stracks[i]->state == TrackState::Tracked) {
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(),
		tracked_stracks_swap.end());

	joint_stracks(this->tracked_stracks, activated_stracks,
		this->tracked_stracks);
	joint_stracks(this->tracked_stracks, refind_stracks, this->tracked_stracks);

	sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < temp_lost_stracks.size(); i++) {
		this->lost_stracks.push_back(temp_lost_stracks[i]);
	}

	sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < temp_removed_stracks.size(); i++) {
		this->removed_stracks.push_back(temp_removed_stracks[i]);
	}
	remove_duplicate_stracks(resa, resb, this->tracked_stracks,
		this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (this->tracked_stracks[i]->is_activated &&
			this->tracked_stracks[i]->tlwh[2] * this->tracked_stracks[i]->tlwh[3] >
			this->min_box_area)
			output_stracks.push_back(this->tracked_stracks[i]);
	}
}

void BYTETracker::joint_stracks(STracks& tlista, STracks& tlistb,
	STracks& results) {
	std::map<int, int> exists;
	for (int i = 0; i < results.size(); i++)
		exists.insert(std::pair<int, int>(results[i]->track_id, 1));

	for (int i = 0; i < tlista.size(); i++) {
		int tid = tlista[i]->track_id;
		if (!exists[tid] || exists.count(tid) == 0) {
			exists[tid] = 1;
			results.push_back(tlista[i]);
		}
	}
	for (int i = 0; i < tlistb.size(); i++) {
		int tid = tlistb[i]->track_id;
		if (!exists[tid] || exists.count(tid) == 0) {
			exists[tid] = 1;
			results.push_back(tlistb[i]);
		}
	}
}

void BYTETracker::sub_stracks(STracks& tlista, STracks& tlistb) {
	std::map<int, std::shared_ptr<STrack>> stracks;
	for (int i = 0; i < tlista.size(); i++)
		stracks.insert(std::pair<int, std::shared_ptr<STrack>>(tlista[i]->track_id,
			tlista[i]));
	for (int i = 0; i < tlistb.size(); i++) {
		int tid = tlistb[i]->track_id;
		if (stracks.count(tid) != 0) stracks.erase(tid);
	}
	tlista.clear();
	for (std::map<int, std::shared_ptr<STrack>>::iterator it = stracks.begin();
		it != stracks.end(); ++it)
		tlista.push_back(it->second);
}

void BYTETracker::remove_duplicate_stracks(STracks& resa, STracks& resb,
	STracks& stracksa,
	STracks& stracksb) {
	std::vector<std::vector<float>> pdist;
	iou_distance(stracksa, stracksb, pdist);
	std::vector<std::pair<int, int>> pairs;
	for (int i = 0; i < pdist.size(); i++) {
		for (int j = 0; j < pdist[i].size(); j++) {
			if (pdist[i][j] < 0.15) {
				pairs.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	std::vector<int> dupa, dupb;
	for (int i = 0; i < pairs.size(); i++) {
		int timep = stracksa[pairs[i].first]->frame_id -
			stracksa[pairs[i].first]->start_frame;
		int timeq = stracksb[pairs[i].second]->frame_id -
			stracksb[pairs[i].second]->start_frame;
		if (timep > timeq)
			dupb.push_back(pairs[i].second);
		else
			dupa.push_back(pairs[i].first);
	}

	for (int i = 0; i < stracksa.size(); i++) {
		std::vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
		if (iter == dupa.end()) {
			resa.push_back(stracksa[i]);
		}
	}

	for (int i = 0; i < stracksb.size(); i++) {
		std::vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
		if (iter == dupb.end()) {
			resb.push_back(stracksb[i]);
		}
	}
}

void BYTETracker::linear_assignment(
	std::vector<std::vector<float>>& cost_matrix, int cost_matrix_size,
	int cost_matrix_size_size, float thresh,
	std::vector<std::vector<int>>& matches, std::vector<int>& unmatched_a,
	std::vector<int>& unmatched_b) {
	if (cost_matrix.size() == 0) {
		for (int i = 0; i < cost_matrix_size; i++) {
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++) {
			unmatched_b.push_back(i);
		}
		return;
	}
	std::vector<int> rowsol;
	std::vector<int> colsol;
	lapjv(cost_matrix, rowsol, colsol, true, thresh);
	for (int i = 0; i < rowsol.size(); i++) {
		if (rowsol[i] >= 0) {
			std::vector<int> match;
			match.push_back(i);
			match.push_back(rowsol[i]);
			matches.push_back(match);
		}
		else {
			unmatched_a.push_back(i);
		}
	}
	for (int i = 0; i < colsol.size(); i++) {
		if (colsol[i] < 0) {
			unmatched_b.push_back(i);
		}
	}
}

void BYTETracker::ious(std::vector<std::vector<float>>& atlbrs,
	std::vector<std::vector<float>>& btlbrs,
	std::vector<std::vector<float>>& results) {
	if (atlbrs.size() * btlbrs.size() == 0) return;

	results.resize(atlbrs.size());
	for (int i = 0; i < results.size(); i++) {
		results[i].resize(btlbrs.size());
	}

	// bbox_ious
	for (int k = 0; k < btlbrs.size(); k++) {
		std::vector<float> ious_tmp;
		float box_area =
			(btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
		for (int n = 0; n < atlbrs.size(); n++) {
			float iw = std::min(atlbrs[n][2], btlbrs[k][2]) -
				std::max(atlbrs[n][0], btlbrs[k][0]) + 1;
			if (iw > 0) {
				float ih = std::min(atlbrs[n][3], btlbrs[k][3]) -
					std::max(atlbrs[n][1], btlbrs[k][1]) + 1;
				if (ih > 0) {
					float ua = (atlbrs[n][2] - atlbrs[n][0] + 1) *
						(atlbrs[n][3] - atlbrs[n][1] + 1) +
						box_area - iw * ih;
					results[n][k] = iw * ih / ua;
				}
				else {
					results[n][k] = 0.0;
				}
			}
			else {
				results[n][k] = 0.0;
			}
		}
	}
}

void BYTETracker::iou_distance(const STracks& atracks, const STracks& btracks,
	std::vector<std::vector<float>>& cost_matrix) {
	if (atracks.size() * btracks.size() == 0) return;

	std::vector<std::vector<float>> atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++) {
		atlbrs.push_back(atracks[i]->tlbr);
	}
	for (int i = 0; i < btracks.size(); i++) {
		btlbrs.push_back(btracks[i]->tlbr);
	}

	std::vector<std::vector<float>> _ious;
	ious(atlbrs, btlbrs, _ious);
	for (int i = 0; i < _ious.size(); i++) {
		std::vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++) {
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}
}

void BYTETracker::lapjv(const std::vector<std::vector<float>>& cost,
	std::vector<int>& rowsol, std::vector<int>& colsol,
	bool extend_cost, float cost_limit, bool return_cost) {
	std::vector<std::vector<float>> cost_c;
	cost_c.assign(cost.begin(), cost.end());

	std::vector<std::vector<float>> cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols) {
		n = n_rows;
	}
	else {
		if (!extend_cost) {
			std::cout << "set extend_cost=True" << std::endl;
			system("pause");
			exit(0);
		}
	}
	if (extend_cost || cost_limit < LONG_MAX) {
		n = n_rows + n_cols;
		cost_c_extended.resize(n);

		for (int i = 0; i < cost_c_extended.size(); i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX) {
			for (int i = 0; i < cost_c_extended.size(); i++) {
				for (int j = 0; j < cost_c_extended[i].size(); j++) {
					cost_c_extended[i][j] = cost_limit / 2.0;
				}
			}
		}
		else {
			float cost_max = -1;
			for (int i = 0; i < cost_c.size(); i++) {
				for (int j = 0; j < cost_c[i].size(); j++) {
					if (cost_c[i][j] > cost_max) cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < cost_c_extended.size(); i++) {
				for (int j = 0; j < cost_c_extended[i].size(); j++) {
					cost_c_extended[i][j] = cost_max + 1;
				}
			}
		}
		for (int i = n_rows; i < cost_c_extended.size(); i++) {
			for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++) {
			for (int j = 0; j < n_cols; j++) {
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}
	double** cost_ptr;
	cost_ptr = new double* [sizeof(double*) * n];
	for (int i = 0; i < n; i++) cost_ptr[i] = new double[sizeof(double) * n];

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	int* x_c = new int[sizeof(int) * n];
	int* y_c = new int[sizeof(int) * n];

	int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
	if (ret != 0) {
		std::cout << "Calculate Wrong!" << std::endl;
		system("pause");
		exit(0);
	}

	double opt = 0.0;
	if (n != n_rows) {
		for (int i = 0; i < n; i++) {
			if (x_c[i] >= n_cols) x_c[i] = -1;
			if (y_c[i] >= n_rows) y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++) {
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++) {
			colsol[i] = y_c[i];
		}

		if (return_cost) {
			for (int i = 0; i < rowsol.size(); i++) {
				if (rowsol[i] != -1) {
					// cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] <<
					// endl;
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost) {
		for (int i = 0; i < rowsol.size(); i++) {
			opt += cost_ptr[i][rowsol[i]];
		}
	}
	for (int i = 0; i < n; i++) {
		delete[] cost_ptr[i];
	}
	delete[] cost_ptr;
	delete[] x_c;
	delete[] y_c;
}
