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

inline void swap(int* a, int* b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

inline double distance_4e(double a1, double a2, double a3, double a4) {

	double max_a = Max(Max(a1, a2), Max(a3, a4));
	double min_a = Min(Min(a1, a2), Min(a3, a4));

	return max_a - min_a;
}


void FindRect(const Size size_, const Mat& P_mat, Point2d& tl_corner, double& width, double& height) {
	
	// top-left , bottom-right, top-right, bottom-left
	Mat corner_mat = (Mat_<double>(3, 4) << 0, size_.width, size_.width,0,     0, size_.height, 0, size_.height,   1,1,1,1);
	Mat P_3x3 = P_mat(Range(0, 3), Range(0, 3));
	Mat projected_corner = P_3x3* corner_mat;
	Point2d new_tl_pt(projected_corner.at<double>(0, 0) / projected_corner.at<double>(2, 0), projected_corner.at<double>(1, 0) / projected_corner.at<double>(2, 0));
	Point2d new_br_pt(projected_corner.at<double>(0, 1) / projected_corner.at<double>(2, 1), projected_corner.at<double>(1, 1) / projected_corner.at<double>(2, 1));
	Point2d new_tr_pt(projected_corner.at<double>(0, 2) / projected_corner.at<double>(2, 2), projected_corner.at<double>(1, 2) / projected_corner.at<double>(2, 2));
	Point2d new_bl_pt(projected_corner.at<double>(0, 3) / projected_corner.at<double>(2, 3), projected_corner.at<double>(1, 3) / projected_corner.at<double>(2, 3));
	double min_x = Min(Min(new_tl_pt.x, new_bl_pt.x), Min(new_tr_pt.x, new_br_pt.x));
	double min_y = Min(Min(new_tl_pt.y, new_bl_pt.y), Min(new_tr_pt.y, new_br_pt.y));
	tl_corner = Point2d(min_x, min_y);
	width = distance_4e(new_tl_pt.x, new_bl_pt.x, new_tr_pt.x, new_br_pt.x);
	height =distance_4e(new_tl_pt.y, new_bl_pt.y, new_tr_pt.y, new_br_pt.y);

}

//tl_corner = top-left corner point
void Translate_img(const Mat& org_img, const Point tl_corner,  Mat& out_img) {
	uchar* data_org = org_img.data;
	uchar* data_out =  out_img.data;
}

void Projecting_img(const Mat& org_img, const Mat& P_mat, Mat& result_img) {
	result_img = org_img.clone();
	Point2d tl_point; 
	double new_width(static_cast<double>(result_img.cols)), new_height(static_cast<double>(result_img.rows));
	FindRect(result_img.size(), P_mat, tl_point, new_width, new_height);
	//--
	// shift all pixel (origin point --> top-left corner )
	//Translate_img()
	//--
	warpPerspective(org_img, result_img, P_mat, Size(new_width, new_height));
#if DRAW_ORG_POSE
	//Mat org_img_rect;
	rectangle(result_img, Rect(Point(0, 0), Point(org_img.cols, org_img.rows)), Scalar(0, 0, 255), 1,4,0);
#endif 

}


cv::Size getDisplaySize() {
	int iMonitorWidth = GetSystemMetrics(SM_CXSCREEN);
	int iMonitorHeight = GetSystemMetrics(SM_CYSCREEN);
	cv::Size display_size = { iMonitorWidth, iMonitorHeight };
	return display_size;
}

cv::Mat& ShowImage(cv::Mat& img, bool img_resize,  int waitKey_val) {

	cv::Mat display_img;
	cv::Size display_size = getDisplaySize();
	if (img_resize && (img.size().width > display_size.width || img.size().height > display_size.height)) {
		double scale_x = static_cast<double>(display_size.width) / static_cast<double>(img.size().width);
		double scale_y = static_cast<double>(display_size.height) / static_cast<double>(img.size().height);
		double scale = min(scale_x, scale_y) * 0.9;
		cv::resize(img, display_img, Size(0,0), scale, scale);
		cv::imshow("display image", display_img);
		cv::waitKey(waitKey_val);
	}
	else {
		cv::imshow("display image", img);
		cv::waitKey(waitKey_val);
	}
	return display_img;
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
				//Mat P_mat = Mat::eye(3, 3, CV_32F);
				Mat P_mat = (Mat_<double>(3, 3) << 1, 0.5, 0.3, 0, 2, 0, 0, 0, 2);


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
		//imshow("w", final_frame_set[f_idx]);
		//waitKey(10);
		ShowImage(final_frame_set[f_idx], true, 10);
	}

#endif

	end_t();
	CN("[DONE]");



	return 0;
}
