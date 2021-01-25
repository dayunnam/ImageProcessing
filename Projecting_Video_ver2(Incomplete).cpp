
/*
Windows
visual studio 2019
C++
opencv4.5.1
*/


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <Windows.h>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <Eigen/Dense>

#define WITH_THREAD 1
#define DRAW_ORG_POSE 0
#define CN(msg) std::cout << msg << std::endl;
#define FN(msg,value) {std::fprintf(stdout,msg, value);fprintf(stdout, "\n");}
#define F(msg,value) {std::fprintf(stdout,msg, value);}
#define input "test_video.mp4"
#define thread_num 5
#define Min(x, y)                   ((x)<(y)?(x):(y)) 
#define Max(x, y)                   ((x)>(y)?(x):(y))
#define Abs(x)                      ((x)>0?(x):(-(x)))  
#define Sqr(x)                      ((x)*(x))      

using namespace cv;
using namespace std;
using namespace Eigen;

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


cv::Size getDisplaySize() {
	int iMonitorWidth = GetSystemMetrics(SM_CXSCREEN);
	int iMonitorHeight = GetSystemMetrics(SM_CYSCREEN);
	cv::Size display_size = { iMonitorWidth, iMonitorHeight };
	return display_size;
}

cv::Mat& ShowImage(cv::Mat& img, bool img_resize, int waitKey_val) {

	cv::Mat display_img;
	cv::Size display_size = getDisplaySize();
	if (img_resize && (img.size().width > display_size.width || img.size().height > display_size.height)) {
		double scale_x = static_cast<double>(display_size.width) / static_cast<double>(img.size().width);
		double scale_y = static_cast<double>(display_size.height) / static_cast<double>(img.size().height);
		double scale = min(scale_x, scale_y) * 0.9;
		cv::resize(img, display_img, Size(0, 0), scale, scale);
		cv::imshow("display image", display_img);
		cv::waitKey(waitKey_val);
	}
	else {
		cv::imshow("display image", img);
		cv::waitKey(waitKey_val);
	}
	return display_img;
}

inline bool InImage(Size img_size, Point point) {
	if (point.x < 0 || point.x > img_size.width) return false;
	if (point.y < 0 || point.y > img_size.height) return false;
	return true;
}

inline int clipping(const int& org_value, const int& min, const int& max) {
	// max = (1 << bitdepth)
	if (org_value < min) return min;
	else if (org_value > max) return  max;
	else return org_value;
}


// https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
inline float
BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
{
	float x2x1, y2y1, x2x, y2y, yy1, xx1;
	x2x1 = x2 - x1;
	y2y1 = y2 - y1;
	x2x = x2 - x;
	y2y = y2 - y;
	yy1 = y - y1;
	xx1 = x - x1;
	return 1.0 / (x2x1 * y2y1) * (
		q11 * x2x * y2y +
		q21 * xx1 * y2y +
		q12 * x2x * yy1 +
		q22 * xx1 * yy1
		);
}

void Projecting_img(const Mat& org_img, const Mat& P_mat, Mat& result_img) {
	if (org_img.empty()) return;
	Mat result_img_ = Mat::zeros(org_img.size(), org_img.type());
	
	double result_width = static_cast<double>(result_img_.cols);
	double result_height = static_cast<double>(result_img_.rows);

	Matrix3d P;
	P << P_mat.at<double>(0, 0), P_mat.at<double>(0, 1), P_mat.at<double>(0, 2),
		P_mat.at<double>(1, 0), P_mat.at<double>(1, 1), P_mat.at<double>(1, 2), 
		P_mat.at<double>(2, 0), P_mat.at<double>(2, 1), P_mat.at<double>(2, 2) ;
	Matrix3d P_inv = P.inverse();

	
	for (int y = 0; y < result_height; y++) {
		for (int x = 0; x < result_width; x++) {
			Vector3d old_point;
			Vector3d cur_point(x, y, 1.0);
			old_point = P_inv * cur_point;
			Point old_pt(static_cast<int>((old_point(0,0)/ old_point(2, 0)) + 0.5), static_cast<int>((old_point(1, 0) / old_point(2, 0)) + 0.5));
			if (InImage(result_img_.size(), old_pt)) {
				result_img_.at<uchar>(y, x) = org_img.at<uchar>(old_pt.y, old_pt.x);
			}
		}
	}
	
	/*
	for (int y = 1; y < result_height-1; y++) {
		for (int x = 1; x < result_width-1; x++) {
			int pt_00 = result_img_.at<uchar>(y-1, x-1);
			int pt_01 = result_img_.at<uchar>(y+1, x-1);
			int pt_10 = result_img_.at<uchar>(y+1, x+1);
			int pt_11 = result_img_.at<uchar>(y-1, x+1);
			
			if (result_img_.at<uchar>(y, x) == 0) {
				result_img.at<uchar>(y, x) = clipping(BilinearInterpolation(pt_00, pt_00, pt_00, pt_00, x - 1, x + 1, y - 1, y + 1, x, y),0, 255);
			}
			else {
				result_img.at<uchar>(y, x) = result_img_.at<uchar>(y, x);
			}

		}
	}*/
	
	result_img = result_img_.clone();
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
				Mat Gray_frame, Projected_frame;
				cvtColor(frame_set[frame_idx], Gray_frame, COLOR_BGR2GRAY);
				
				//Mat P_mat = Mat::eye(3, 3, CV_32F);
				Mat P_mat = (Mat_<double>(3, 3) << 1, 0.0, 0.0, 0, 1,0, 0, 0, 0, 1,0);
				Projecting_img(Gray_frame, P_mat, Projected_frame);
				frame_subset[thread_idx].emplace_back(Projected_frame);
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
		//imshow("w", final_frame_set[f_idx]);
		//waitKey(10);
		ShowImage(final_frame_set[f_idx], true, 10);
	}

#endif

	end_t();
	CN("[DONE]");

	return 0;
}

