
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
    cv::String result_name_ = argv[2];
    float scale_f = -1.0; 
    if (argv[3] != NULL) scale_f = std::stof(argv[3]);

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
    std::vector<cv::Mat> org_img_set(num_images);
    std::vector<cv::Mat> scaled_img_set(num_images);
    std::cout << "The number of pictures : " << num_images << std::endl;
    if (scale_f > 0.0) std::cout << "scale factor : " << scale_f << std::endl;
    
    for (int i = 0; i < num_images; ++i)
    {
        org_img_set[i] = cv::imread(img_names[i]);

        if (org_img_set[i].empty())
        {
            std::cout << "Can't open image " << img_names[i] << std::endl;
            getchar();
            return -1;
        }
        if (scale_f > 0) {
            resize(org_img_set[i], scaled_img_set[i], cv::Size(), scale_f, scale_f);
            std::string buff = img_names[i].substr(0, img_names[i].length()-4) + "_s.png";
            std::cout << "original image size : " << org_img_set[i].size()<< std::endl;
            std::cout << "scaled image size : " << scaled_img_set[i].size()<< std::endl;
            cv::imwrite(buff, scaled_img_set[i]);
        }
    }
    std::cout << "[DONE]" << std::endl;
    return 0;
}

