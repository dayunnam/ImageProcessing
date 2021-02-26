
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stdout, "ERROR::No input dataset\nIn order to run the program type: 'Read_picture_from_directory.exe [the path of input data] [the name of output] [(optional)scale factor]'\n");
        return -1;
    }
    
    cv::String path_ = argv[1];
    std::vector<cv::String> img_names;
    try
    {
        cv::glob(path_, img_names, true);
        if (img_names.size() < 2)
            throw   img_names;
    }
    catch (cv::Exception)
    {
        std::cout << "There are no folder named " << path_ << " in tihs directory" << std::endl;
        getchar();
        return -1;
    }
    catch (std::vector<cv::String> img_names_)
    {
        std::cout << "Need more images in " << path_ << std::endl;
        getchar();
        return -1;
    }
    int num_images = static_cast<int>(img_names.size());

    std::cout << "The number of pictures : " << num_images << std::endl;
    

    return 0;
}
