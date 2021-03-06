// visualization of 2D filter
/*
Windows
visual studio 2019
C++
opencv4.5.1
Eigen 3.3.9
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <math.h>

#define CN(a) std::cout << a << std::endl;
#define result "result.png"
#define PI 3.14159265358
#define Spr(x) ((x)*(x))
using namespace cv;
using namespace Eigen;
using namespace std;

//2D Gaussian normal distribution
double& Normal_Func(const double x1, const double x2, const Vector2d mean_, const Matrix2d cov, const double scale) {
	Matrix2d inv_cov = cov.inverse();
	Vector2d pos(x1 - mean_(0), x2 - mean_(1));
	double temp = pos.transpose() * inv_cov * pos;
	double N = scale * 1.0 / (2.0 * PI) * exp(-temp / 2.0);
	return N;
}

//Draw Distribution
Mat Draw_Distribution(const Size mat_size, double ex, double ey, double exy) {
	Mat distribution = Mat::zeros(mat_size, CV_64FC1);
	Vector2d center(mat_size.width / 2 - 1, mat_size.height / 2 - 1);
	Matrix2d cov;
	cov << ex, exy,
		exy, ey;
	for (int p_y = 0; p_y < mat_size.height; p_y++) {
		for (int p_x = 0; p_x < mat_size.width; p_x++) {
			distribution.at<double>(p_y, p_x) = Normal_Func(p_x, p_y, center, cov, 5);
			//CN(distribution.at<double>(p_y, p_x));
		}
	}
	return distribution;
}


int main() {
	double ex = 1.0, ey = 1.0, exy = 1.0;
	for (int n = 0; n < 500; n++) {
		ex *= 1.02;
		ey *= 1.02;
		exy *= 0.99;
		Mat img = Draw_Distribution(Size(512, 512), ex, ey, exy);
		imshow("Normal Distribution", img);
		waitKey(5);
	}
	CN("[DONE]");

	return 0;
}
