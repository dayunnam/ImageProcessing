#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
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
#define result "result.mp4"
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
				frame_subset[thread_idx].emplace_back(frame_set[frame_idx]);
			}

			}));
	}
	for (int thread_idx2 = 0; thread_idx2 < thread_num; thread_idx2++) Threads[thread_idx2].join();
	Threads.clear();

	for (int t_idx = 0; t_idx < thread_num; t_idx++) {
		for (int idx = 0; idx < frame_subset[t_idx].size(); idx++) {
			imshow("cur_frame", frame_subset[t_idx][idx]);
			waitKey(10);
		}
	}
#else 

#endif

	

	end_t();
	CN("[DONE]");

	
	
	return 0;
}
