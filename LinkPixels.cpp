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
#define input "test.png"
#define thread_num 5

using namespace cv;
using namespace std;


static struct std::chrono::system_clock::time_point start_time;
struct pixel_st;
struct edge_st;

using pixel_ptr = shared_ptr<pixel_st>;
using edge_ptr = shared_ptr<edge_st>;


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

struct pixel_st {
	int col_r;
	int col_g;
	int col_b;
	int pos_x;
	int pos_y;
	int pos_z;
	int alpha;
};

struct edge_st {
	double weight;
	pixel_ptr vertex1;
	pixel_ptr vertex2;
};

void display_edge_info(edge_ptr edge_){
	CN("-------------");
	CN("Vertex 1 info ");
	FN("B : %d", edge_->vertex1->col_b);
	FN("G : %d", edge_->vertex1->col_g);
	FN("R : %d", edge_->vertex1->col_r);
	FN("X : %d", edge_->vertex1->pos_x);
	FN("Y : %d", edge_->vertex1->pos_y);
	FN("Z : %d", edge_->vertex1->pos_z);
	CN("Vertex 2 info ");
	FN("B : %d", edge_->vertex2->col_b);
	FN("G : %d", edge_->vertex2->col_g);
	FN("R : %d", edge_->vertex2->col_r);
	FN("X : %d", edge_->vertex2->pos_x);
	FN("Y : %d", edge_->vertex2->pos_y);
	FN("Z : %d", edge_->vertex2->pos_z);
	CN("");
	FN("weight : %f" , edge_->weight);
	CN("-------------");
}
edge_ptr setEdge(double (*weight_function)(pixel_ptr,pixel_ptr), pixel_ptr pix1, pixel_ptr pix2) {
	edge_ptr edge = std::make_shared<edge_st>();
	edge->weight = (*weight_function)(pix1, pix2);
	edge->vertex1 = pix1;
	edge->vertex2 = pix2;
	return edge;
}

double weight_func(pixel_ptr pix1, pixel_ptr pix2) {
	double w = static_cast<double>((pix1->col_b - pix2->col_b)+ (pix1->col_g - pix2->col_g)+ (pix1->col_r - pix2->col_r)+ (pix1->pos_x - pix2->pos_x)+ (pix1->pos_y - pix2->pos_y)+ (pix1->pos_z - pix2->pos_z));
	return [](double val) {if (val > 0) return val; else return -val; } (w);
}

vector<pixel_ptr> CpoyPixelVector(const Mat& input_img) {
	
	uchar* data_img = input_img.data;
	int width = input_img.cols, height = input_img.rows;
	vector<pixel_ptr> graph_img(width * height);
	for (int p_y = 0; p_y < height; p_y++) {
		for (int p_x = 0; p_x < width; p_x++) {
			
			int p_idx = p_y * width + p_x;
			graph_img[p_idx] = make_shared<pixel_st>();
			graph_img[p_idx]->col_b = *(data_img + p_idx * 3);
			graph_img[p_idx]->col_g = *(data_img + p_idx * 3 + 1);
			graph_img[p_idx]->col_r = *(data_img + p_idx * 3 + 2);
			graph_img[p_idx]->pos_x = p_x;
			graph_img[p_idx]->pos_y = p_y;
			graph_img[p_idx]->pos_z = 1;
			
		}
	}
	return graph_img;
}


int main() {
	
	int pix1_idx = 8;
	int pix2_idx = 9;

	start_t();
	Mat org_img = imread(input);
	vector<pixel_ptr> pixel_set = CpoyPixelVector(org_img);
	
	
	edge_ptr edge = setEdge(weight_func, pixel_set[pix1_idx], pixel_set[pix2_idx]);
	display_edge_info(edge);
	end_t();
	CN("[DONE]");

	return 0;
}
