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
#define input "pic.png"
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

edge_ptr CreateEdge(double (*weight_function)(pixel_ptr,pixel_ptr), pixel_ptr pix1, pixel_ptr pix2) {
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

vector<pixel_ptr> CopyPixelVector(const Mat& input_img) {
	
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


__int64 getSizeEdge(int rows, int cols) {
	__int64 mid = 4 * (rows - 1) * (cols - 2);
	__int64 col_first = 3 * (rows - 1);
	__int64 col_last = 2 * (rows - 1);
	__int64 row_last = cols - 1;
	return  mid + col_first+ col_last+ row_last;
}

vector<edge_ptr>  GenEdgeVector(vector<pixel_ptr> pixel_set, int rows, int cols){
	__int64 size_vec = pixel_set.size();
	vector<edge_ptr> edge_set(getSizeEdge(rows,cols));
	int e_idx = 0;
	for (int p_y = 0; p_y < rows; p_y++) {
		for (int p_x = 0; p_x < cols; p_x++) {
			__int64 p_idx = p_y * cols + p_x;
			edge_set[e_idx] = make_shared<edge_st>();
			if (p_x != cols - 1) edge_set[e_idx++] = CreateEdge(weight_func, pixel_set[p_idx], pixel_set[p_idx + 1]);
			if (p_y != rows - 1) edge_set[e_idx++] = CreateEdge(weight_func, pixel_set[p_idx], pixel_set[p_idx + cols]);
			if (p_y != rows - 1 && p_x != cols - 1) edge_set[e_idx++] = CreateEdge(weight_func, pixel_set[p_idx], pixel_set[p_idx + cols + 1]);
			if (p_y != rows - 1 && p_x != 0) edge_set[e_idx++] = CreateEdge(weight_func, pixel_set[p_idx], pixel_set[p_idx + cols - 1]);

			
		}
	}
	return edge_set;
}

class graph_pix {
public:
	typedef void* vertex;
	typedef void* link;
	
	struct unit {
		vertex node;
		link edge;
	};
	vector<unit> graph;
	
};

void processing(const vector<edge_ptr>& edge_set, Mat& out_img) {
	pixel_ptr pixel1;
	pixel_ptr pixel2;
	int width = out_img.cols;
	uchar* out_data = out_img.data;
	for (const auto& edge : edge_set) {
		int idx = 3 * (edge->vertex1->pos_x + edge->vertex1->pos_y * width);
		uchar intensity = static_cast<uchar>(5 * (edge->weight));
		*(out_data + idx) += intensity;
		*(out_data + idx + 1) += intensity;
		*(out_data + idx + 2) += intensity;
	
	}
}

int main() {
	
	int pix1_idx = 8;
	int pix2_idx = 9;

	start_t();
	Mat org_img = imread(input);
	vector<pixel_ptr> pixel_set = CopyPixelVector(org_img);
	vector<edge_ptr>  edge_set= GenEdgeVector(pixel_set, org_img.rows, org_img.cols);
	FN("Pixel Count : %d", pixel_set.size());
	FN("Edge Count : %d", edge_set.size());
	Mat out_img = Mat::zeros(org_img.size(),org_img.type());
	processing(edge_set, out_img);
	//edge_ptr edge = CreateEdge(weight_func, pixel_set[pix1_idx], pixel_set[pix2_idx]);
	//display_edge_info(edge);

	imshow("w", out_img);
	waitKey(0);
	end_t();
	CN("[DONE]");

	return 0;
}
