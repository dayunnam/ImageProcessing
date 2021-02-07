#include <opencv2/opencv.hpp>
using namespace cv;
//original code https://learnopencv.com/tag/poisson-image-editing/
int main() {
    
    Mat src = imread("Inpaint_test/hand.png");
    Mat dst = imread("Inpaint_test/eye.png");
    Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
    Point poly[1][7];
    poly[0][0] = Point(40, 40);
    poly[0][1] = Point(80, 40);
    poly[0][2] = Point(80, 80);
    poly[0][3] = Point(40, 80);

    const Point* polygons[1] = { poly[0] };
    int num_points[] = { 7 };
    fillPoly(src_mask, polygons, num_points, 1, Scalar(255, 255, 255));
    Point center(50, 50);
    Mat output;
    seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);
    imwrite("Inpaint_test/mask.png", src_mask);
    imwrite("Inpaint_test/opencv-seamless-cloning-example.png", output);

    return 0;
}
