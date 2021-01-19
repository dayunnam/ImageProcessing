#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <iostream>
#include <chrono>

#define CN(a) std::cout << a << std::endl;
#define result "result.png"
using namespace cv;
using namespace std;

int add(const int& a, const int& b) {
	return a + b;
}

int multiply(const int& a, const int& b) {
	return a * b;
}


int clipping(const int& org_value, const int& bitdepth) {
	if (org_value < 0) return 0;
	else if (org_value >= (1 << bitdepth)) return (1 << bitdepth) - 1;
	else return org_value;
}

int main() {
	Mat img = imread("pic.png");
	Mat c_img(img);
	int (*function_pt)(const int&, const int&);
	function_pt = &clipping;
	const uchar* data = img.data;
	uchar* c_data = c_img.data;

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int idx = 3 * (y * img.cols + x);
			*(c_data + idx) =  (*function_pt)(*(data + idx) >> 1, 8);
			*(c_data + ++idx) = (*function_pt)(*(data + idx) >> 1, 8);
			*(c_data + ++idx) =  (*function_pt)(*(data + idx) >> 1, 8);
		}
	}

	imwrite(result, c_img);

	return 0;
}
