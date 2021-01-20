#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#define WITH_THREAD 1

#define CN(msg) std::cout << msg << std::endl;
#define FN(msg,value) {std::fprintf(stdout,msg, value);fprintf(stdout, "\n");}
#define F(msg,value) {std::fprintf(stdout,msg, value);}
#define input "test_video.mp4"
#define thread_num 5

using namespace cv;
using namespace std;


static struct std::chrono::system_clock::time_point start_time;


static void start_t() {
	start_time = std::chrono::system_clock::now();
}

static double end_t() {
	std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
	std::chrono::duration<double> sec = end_time - start_time;
	double elapsed = sec.count();
	std::cout << "Elapsed time : " << elapsed << " sec" << std::endl;

	return elapsed;
}

void filtering_img(const Mat& org_img, const Mat& kernel, Mat& result_img) {
	filter2D(org_img, result_img, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
}

void Projecting_img(const Mat& org_img, const Mat& P_mat, Mat& result_img) {
	result_img = org_img.clone();
	warpPerspective(org_img, result_img, P_mat, result_img.size());
}

int main() {
	start_t();
	VideoCapture org_video(input);
	if (!org_video.isOpened()) {
		CN(input << " can not open");
		return -1;
	}
	const int frame_num = org_video.get(CAP_PROP_FRAME_COUNT);
	const int frame_width = org_video.get(CAP_PROP_FRAME_WIDTH);
	const int frame_height = org_video.get(CAP_PROP_FRAME_HEIGHT);

	vector<Mat> frame_set(frame_num);


	Mat frame;
	for (int idx = 0; idx < frame_num; idx++) {
		org_video >> frame;
		if (frame.empty())
			break;
		frame.copyTo(frame_set[idx]);
	}


#if WITH_THREAD
	vector<vector<Mat>> frame_subset(thread_num);
	vector<thread> Threads;
	for (unsigned int thread_idx = 0; thread_idx < thread_num; thread_idx++) {
		Threads.push_back(thread([&, thread_idx]() {

			int max_frame = static_cast<int>((thread_idx + 1) * frame_num / static_cast<double>(thread_num));
			int base_frame = static_cast<int>((thread_idx) * static_cast<double>(frame_num) / static_cast<double>(thread_num));
			frame_subset[thread_idx].reserve(max_frame - base_frame);
			//std::fprintf(stdout, "thread# : %d,  frame# %d ~ %d [total %d frame]\n", thread_idx, base_frame, max_frame, frame_num);
			Mat frame;
			for (int frame_idx = base_frame; frame_idx < max_frame; frame_idx++) {

				Mat filtered_img_;
				Mat P_mat = Mat::eye(3, 3, CV_32F);


				Projecting_img(frame_set[frame_idx], P_mat, filtered_img_);
				frame_subset[thread_idx].emplace_back(filtered_img_);
			}

			}));
	}
	for (int thread_idx2 = 0; thread_idx2 < thread_num; thread_idx2++) Threads[thread_idx2].join();
	Threads.clear();

	vector<Mat> final_frame_set(frame_num);
	int frame_idx = 0;
	for (int t_idx = 0; t_idx < thread_num; t_idx++) {
		for (int idx = 0; idx < frame_subset[t_idx].size(); idx++) {
			if (frame_idx >= frame_num) break;
			frame_subset[t_idx][idx].copyTo(final_frame_set[frame_idx]);
			frame_idx++;
		}
	}

	for (int f_idx = 0; f_idx < final_frame_set.size(); f_idx++) {
		imshow("w", final_frame_set[f_idx]);
		waitKey(10);
	}

#endif

	end_t();
	CN("[DONE]");



	return 0;
}
