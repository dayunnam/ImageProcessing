#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#define CN(a) std::cout << a << std::endl;
#define input "test_video.mp4"
#define result "result.mp4"

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
	VideoCapture org_video(input);

	if (!org_video.isOpened()) {
		CN(input << " can not open");
		return -1;
	}

	const int frame_num =  org_video.get(CAP_PROP_FRAME_COUNT);
	const int frame_width =  org_video.get(CAP_PROP_FRAME_WIDTH);
	const int frame_height =  org_video.get(CAP_PROP_FRAME_HEIGHT);

	//CN("frame_num : " << frame_num);
	//CN("frame_width : " << frame_width);
	//CN("frame_height : " << frame_height);
	
	Mat frame;
	vector<Mat> frame_set(frame_num);
	for (int idx = 0; idx < frame_num; idx++) {
		org_video >> frame;
		if (frame.empty())
			break;

		frame.copyTo(frame_set[idx]);

		imshow("cur_frame", frame_set[idx]);
		waitKey(10);
	}
	CN("[DONE]");
	return 0;
}
