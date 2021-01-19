#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <stdio.h>
#include <iostream>
#include <chrono>

#define CN(a) std::cout << a << std::endl;
#define input "pic.png"
#define result "result.png"
#define FD_mode "FAST" //Feature Detection mode
/*
FREAK
AKAZE 
ORB
FAST
BRISK
*/
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


int main() {
	start_t();
	Mat img = imread(input);

	if (img.empty()) {
		CN("[ERROR] Empty imput " << input);
		return -1;
	}
	
	vector<KeyPoint> keypoint_;
	Mat descriptor_;
	if (FD_mode == "AKAZE") {
		Ptr<AKAZE> detector = AKAZE::create();
		detector->detectAndCompute(img, noArray(), keypoint_, descriptor_);
	}
	else if (FD_mode == "SIFT") {
		Ptr<SIFT> detector = SIFT::create();
		detector->detectAndCompute(img, noArray(), keypoint_, descriptor_);
	}
	else if (FD_mode == "ORB") {
		Ptr<ORB> detector = ORB::create();
		detector->detectAndCompute(img, noArray(), keypoint_, descriptor_);
	}
	else if (FD_mode == "BRISK") {
		Ptr<BRISK> detector = BRISK::create();
		detector->detectAndCompute(img, noArray(), keypoint_, descriptor_);
	}
	else if (FD_mode == "FAST") {
		Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
		detector->detect(img, keypoint_, noArray());
	}
	else {
		std::cout << "[ERROR] There is not feature detection mode " << FD_mode << std::endl;
		return -1;
	}
	Mat feature_img;
	CN("the number of keypoints : " << keypoint_.size());
	//draw keypoints on image

	drawKeypoints(img, keypoint_, feature_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string result_name = result;
	imwrite(result_name, feature_img);
	end_t();
	CN("[DONE]");
	return 0;
}
