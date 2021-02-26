
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#endif

std::string features_type = "orb";
float match_conf = 0.5f;
std::string matcher_type = "homography";
std::string estimator_type = "homography";
bool try_cuda = false;
int range_width = -1;
float conf_thresh = 1.f;
bool show_match = true;


class Img_info {
public:
    cv::Mat Image;
    std::string FileName;
    cv::Mat H;
    cv::Mat K;
    cv::Mat R;
    cv::Mat T;
    std::vector<cv::KeyPoint> features;
    int img_idx;
    int min_z_val; // minimium depth value
    int max_z_val; // minimium depth value
};


bool pose_estimatoin(std::vector<cv::Mat>& img_set, std::vector<Img_info>& Image_info_) {
    int num_img = img_set.size();
    cv::Ptr<cv::Feature2D> finder;

    std::vector<cv::detail::ImageFeatures> feature_set(num_img);

    if (features_type == "orb")
    {
        finder = cv::ORB::create(2000);
    }
    else if (features_type == "akaze")
    {
        finder = cv::AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = cv::xfeatures2d::SURF::create(400);
    }
    else if (features_type == "sift") {
        finder = cv::SIFT::create();
    }
#endif
    else
    {
        std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    for (int pic_idx = 0; pic_idx < num_img; pic_idx++) {
        cv::detail::computeImageFeatures(finder, img_set[pic_idx], feature_set[pic_idx]);
        feature_set[pic_idx].img_idx = pic_idx;
    }
    
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher;

    if (matcher_type == "affine")
        matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width == -1)
        matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    (*matcher)(feature_set, pairwise_matches);
    matcher->collectGarbage();

    std::vector<int> indices = leaveBiggestComponent(feature_set, pairwise_matches, conf_thresh);
    int num_images = static_cast<int>(indices.size());

    Image_info_.resize(num_images);

    
    if (num_images < 2)
    {
        std::cout << "Need more images" << std::endl;
        return false;
    }

    for (size_t i = 0; i < num_images; i++) {
        Image_info_[i].Image = img_set[indices[i]];
        Image_info_[i].img_idx = indices[i];
        Image_info_[i].features = feature_set[i].keypoints;
    }

    cv::Ptr<cv::detail::Estimator> estimator;
    if (estimator_type == "affine")
        estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    else
        estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    std::vector<cv::detail::CameraParams> cameras;
    if (!(*estimator)(feature_set, pairwise_matches, cameras))
    {
        std::cout << "Homography estimation failed.\n";
        return false;
    }

    std::vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        //std::cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R;
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    double focal_init, cx_init, cy_init;
    if (focals.size() % 2 == 1)
        focal_init = focals[focals.size() / 2];
    else
        focal_init = focals[focals.size() / 2 - 1] + focals[focals.size() / 2] * 0.5f;
    cx_init = img_set.front().cols / 2;
    cy_init = img_set.front().rows / 2;

    
    std::vector<std::pair<uint, uint>> match_pair;
    std::vector<std::vector<cv::DMatch>> match_inlier;

    for (size_t i = 0; i < num_images; i++){
        for (size_t j = i + 1; j < num_images; j++){
            int pair_idx = i * num_images + j;

            if (pairwise_matches[pair_idx].confidence < conf_thresh) continue;

            std::vector<cv::DMatch> matches_ = pairwise_matches[pair_idx].matches;
            std::vector<cv::DMatch> inlier;
            std::vector<cv::Point2d> src, dst;
            for (auto itr = matches_.begin(); itr != matches_.end(); itr++)
            {
                src.push_back(Image_info_[i].features[itr->queryIdx].pt);
                dst.push_back(Image_info_[i].features[itr->trainIdx].pt);
            }
            for (size_t k = 0; k < pairwise_matches[pair_idx].matches.size() && pairwise_matches[pair_idx].matches.size() > 5; ++k)
            {
                if (!pairwise_matches[pair_idx].inliers_mask[k])
                    continue;
                inlier.push_back(matches_[k]);
            }
            
            fprintf(stdout, "Image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), matches_.size());
            
            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);
            
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(Image_info_[i].Image, Image_info_[i].features, Image_info_[j].Image, Image_info_[j].features, inlier,
                    match_image, cv::Scalar::all(-1), cv::Scalar::all(-1));
                cv::imshow("Feature and Matches", match_image);
                cv::waitKey(2000);
            }
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stdout, "ERROR::No input dataset\nIn order to run the program type: 'Matching_Multi-View.exe [the path of input data] [the name of output] [(optional)scale factor] [(optional)the path of output]'\n");
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
            //std::cout << "original image size : " << org_img_set[i].size()<< std::endl;
            //std::cout << "scaled image size : " << scaled_img_set[i].size()<< std::endl;
            //cv::imwrite(buff, scaled_img_set[i]);
        }
    }
    std::vector<Img_info> Image_info;
    pose_estimatoin(scaled_img_set, Image_info);

    std::cout << "[DONE]" << std::endl;
    return 0;
}
