
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
#include <opencv2/core/eigen.hpp>


#define WITH_THREAD 1
#define DRAW_ORG_POSE 0
#define CN(msg) std::cout << msg << std::endl;
#define FN(msg,value) {std::fprintf(stdout,msg, value);fprintf(stdout, "\n");}
#define F(msg,value) {std::fprintf(stdout,msg, value);}
#define input "pic.png"
#define thread_num 5
#define Min(x, y)                   ((x)<(y)?(x):(y)) 
#define Max(x, y)                   ((x)>(y)?(x):(y))
#define Abs(x)                      ((x)>0?(x):(-(x)))  
#define Sqr(x)                      ((x)*(x))      

using namespace cv;
using namespace std;
//using namespace Eigen;

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



int main() {
	start_t();
	Mat result_mat;
	Mat cv_mat = imread(input);
	if (cv_mat.empty()) {
		CN(input << " can not open");
		return -1;
	}

	//Applying reshape()
	//channel : 3 channels (color) --> 1 channels
	//Size : cols x rows = cols*3 x rows   
	Mat cv_mat_1ch = cv_mat.reshape(1, 0);
	 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_eigen_mat;
	//Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> uc_eigen_mat;

	cv::cv2eigen(cv_mat_1ch, d_eigen_mat);
	cv::eigen2cv(d_eigen_mat, result_mat);
	
	
	result_mat = result_mat.reshape(cv_mat.channels(), 0)*(1.0/255.0);
	ShowImage(result_mat, true, 10);
	waitKey(0);
	end_t();
	CN("[DONE]");

	return 0;
}

