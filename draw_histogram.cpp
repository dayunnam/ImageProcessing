/*
2021-02-01
- opencv4.5.1
- x64-windows
- visual studio 2019

original code : https://dsp.stackexchange.com/questions/45467/why-entropy-is-undefined-in-low-contrast-image
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;
using namespace std;

#define CN(msg) std::cout << msg << std::endl;
#define FN(msg,value) {std::fprintf(stdout,msg, value);fprintf(stdout, "\n");}
#define F(msg,value) {std::fprintf(stdout,msg, value);}
#define input "1.png"


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


float entropy(const Mat& seq, const Size& size, const int& index)
{
    int cnt = 0;
    float entr = 0;
    float total_size = size.height * size.width; //total size of all symbols in an image
    for (int i = 0; i < index; i++)
    {
        float sym_occur = (float)seq.at<uchar>(0, i); //the number of times a sybmol has occured
        if (sym_occur > 0) //log of zero goes to infinity
        {
            cnt++;
            entr += (sym_occur / total_size) * (log2(total_size / sym_occur));
        }
    }
    return entr;
}

void imhist(const Mat& image, int histogram[])
{
    // initialize all intensity values to 0
    for (int i = 0; i < 256; i++)   histogram[i] = 0;
  
    // calculate the no of pixels for each intensity values
    for (int y = 0; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++)
            histogram[(int)image.at<uchar>(y, x)]++;
}

Mat histDisplay(int histogram[], const char* name)
{
    int hist[256];
    for (int i = 0; i < 256; i++)    hist[i] = histogram[i];
    
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    int max = hist[0];
    for (int i = 1; i < 256; i++) {
        if (max < hist[i])  max = hist[i];
    }

    // normalize the histogram between 0 and histImage.rows

    for (int i = 0; i < 256; i++) {
        hist[i] = ((double)hist[i] / max) * histImage.rows;
    }

    // draw the intensity line for histogram
    for (int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w * (i), hist_h),
            Point(bin_w * (i), hist_h - hist[i]),
            Scalar(0, 0, 0));
    }

    // display histogram
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histImage);
    waitKey(0);
    return histImage;
}

int main() {
    start_t();
    Mat image = imread(input, IMREAD_GRAYSCALE);
    int hist[256];
    imhist(image, hist);
    Mat histImg = histDisplay(hist, "hist");
    float en = entropy(histImg, image.size(), 256);
    CN("entropy : " << en);
    end_t();
    CN("[DONE]");
}
