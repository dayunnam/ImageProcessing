#pragma once

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
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <cmath>
#include <math.h>

double epsilon_ = 0.0000001;
double pi_ = 3.1415926535;
std::string features_type = "orb";
float match_conf = 0.5f;
std::string matcher_type = "homography";
std::string estimator_type = "homography";
bool try_cuda = false;
int range_width = -1;
float conf_thresh = 1.f;
bool show_match = false;
double distance_scale = 1e-3; // meters

struct ReprojectionError7DOF
{
    ReprojectionError7DOF(const cv::Point2d& _x, const cv::Point2d& _c) : x(_x), c(_c) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        const T& f = camera[6];
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError7DOF, 2, 7, 3>(new ReprojectionError7DOF(_x, _c)));
    }

private:
    const cv::Point2d x;
    const cv::Point2d c;
};

struct ReprojectionError
{
    ReprojectionError(const cv::Point2d& _x, double _f, const cv::Point2d& _c) : x(_x), f(_f), c(_c) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, double _f, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(_x, _f, _c)));
    }

private:
    const cv::Point2d x;
    const double f;
    const cv::Point2d c;
};

class SFM
{
public:

    typedef cv::Vec<double, 9> Vec9d;

    typedef std::unordered_map<uint, uint> VisibilityGraph;

    static inline uint genKey(uint cam_idx, uint obs_idx) { return ((cam_idx << 16) + obs_idx); }

    static inline uint getCamIdx(uint key) { return ((key >> 16) & 0xFFFF); }

    static inline uint getObsIdx(uint key) { return (key & 0xFFFF); }

    static bool addCostFunc7DOF(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError7DOF::create(x, cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }

    static bool addCostFunc6DOF(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError::create(x, camera[6], cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }
};

class Img_info {
public:
    cv::Mat Image;
    std::string FileName;
    double focal; //focal
    double c_x, c_y; //c_x ,c_y
    double yaw, pitch, roll; //rotation
    double pos0, pos1, pos2; //position
    std::vector<cv::KeyPoint> features;
    int img_idx;
    double min_z_val; // minimium depth value
    double max_z_val; // maximum depth value
};

void RotationMatrixToEulerAngles(const cv::Matx33d& R_, double& yaw, double& pitch, double& roll) {
    cv::Matx33d permute(0, 0, 1.0, -1.0, 0, 0, 0, -1.0, 0);
    cv::Matx33d A = permute * R_;
    cv::Matx33d B = A * permute.t();
    cv::Matx33d R = B.t();

    yaw = 0.0;
    pitch = 0.0;
    roll = 0.0;

    if (std::fabs(R(0, 0)) < epsilon_ && std::fabs(R(1, 0)) < epsilon_) {
        yaw = std::atan2(R(1, 2), R(0, 2));
        if (std::fabs(R(2, 0)) < epsilon_) pitch = pi_ / 2.0;
        else pitch = -pi_ / 2.0;
        roll = 0.0;
    }
    else {
        yaw = std::atan2(R(1, 0), R(0, 0));
        if (std::fabs(R(0, 0)) < epsilon_) pitch = std::atan2(-R(2, 0), R(1, 0) / std::sin(yaw));
        else pitch = std::atan2(-R(2, 0), R(0, 0) / std::cos(yaw));

        roll = atan2(R(2, 1), R(2, 2));
    }

    yaw *= (180.0 / pi_);
    pitch *= (180.0 / pi_);
    roll *= (180.0 / pi_);
}

void TranslationToPosition(const cv::Vec3d& tvec, double scale, double& pos_0, double& pos_1, double& pos_2) {
    pos_0 = scale * tvec[2];
    pos_1 = -scale * tvec[0];
    pos_2 = -scale * tvec[1];
}

std::vector<bool> maskNoisyPoints(std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<SFM::Vec9d>& views, const SFM::VisibilityGraph& visibility, double reproj_error2)
{
    std::vector<bool> is_noisy(Xs.size(), false);
    if (reproj_error2 > 0)
    {
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            cv::Point3d& X = Xs[visible->second];
            if (X.z < 0) continue;
            int img_idx = SFM::getCamIdx(visible->first), pt_idx = SFM::getObsIdx(visible->first);
            const cv::Point2d& x = xs[img_idx][pt_idx].pt;
            const SFM::Vec9d& view = views[img_idx];

            // Project the given 'X'
            cv::Vec3d rvec(view[0], view[1], view[2]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Point3d X_p = R * X + cv::Point3d(view[3], view[4], view[5]);
            const double& f = view[6], & cx = view[7], & cy = view[8];
            cv::Point2d x_p(f * X_p.x / X_p.z + cx, f * X_p.y / X_p.z + cy);

            // Calculate distance between 'x' and 'x_p'
            cv::Point2d d = x - x_p;
            if (d.x * d.x + d.y * d.y > reproj_error2) is_noisy[visible->second] = true;
        }
    }
    return is_noisy;
}

bool pose_estimatoin(const std::vector<cv::Mat>& img_set, const std::vector<std::string>& img_name, std::vector<Img_info>& Image_info_) {
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
    /*
    else if (features_type == "sift") {
        finder = cv::SIFT::create();
    }*/
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
        Image_info_[i].FileName = img_name[indices[i]];
        Image_info_[i].img_idx = indices[i];
        Image_info_[i].features = feature_set[i].keypoints;
    }

    cv::Ptr<cv::detail::Estimator> estimator;
    if (estimator_type == "affine")
        estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    else
        estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    std::vector<cv::detail::CameraParams> cameras_;
    if (!(*estimator)(feature_set, pairwise_matches, cameras_))
    {
        std::cout << "Homography estimation failed.\n";
        return false;
    }

    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        //std::cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R;
        focals.push_back(cameras_[i].focal);
    }

    sort(focals.begin(), focals.end());
    double focal_init = 500, cx_init = -1, cy_init = -1;
    double Z_init = 2, Z_limit = 100, ba_loss_width = 9;
    int ba_num_iter = 200;

    if (focals.size() % 2 == 1)
        focal_init = focals[focals.size() / 2];
    else
        focal_init = focals[focals.size() / 2 - 1] + focals[focals.size() / 2] * 0.5f;
    cx_init = img_set.front().cols / 2;
    cy_init = img_set.front().rows / 2;


    std::vector<std::pair<uint, uint>> match_pair;
    std::vector<std::vector<cv::DMatch>> match_inlier;

    for (size_t i = 0; i < num_images; i++) {
        for (size_t j = i + 1; j < num_images; j++) {
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
    if (match_pair.size() < 1) return false;

    // 1) Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, focal_init, cx_init, cy_init));

    // 2) Initialize 3D points and build a visibility graph
    std::vector<cv::Point3d> Xs;
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (size_t m = 0; m < match_pair.size(); m++)
    {
        for (size_t in = 0; in < match_inlier[m].size(); in++)
        {
            const uint& cam1_idx = match_pair[m].first, & cam2_idx = match_pair[m].second;
            const uint& x1_idx = match_inlier[m][in].queryIdx, & x2_idx = match_inlier[m][in].trainIdx;
            const uint key1 = SFM::genKey(cam1_idx, x1_idx), key2 = SFM::genKey(cam2_idx, x2_idx);
            auto visit1 = xs_visited.find(key1), visit2 = xs_visited.find(key2);
            if (visit1 != xs_visited.end() && visit2 != xs_visited.end())
            {
                // Remove previous observations if they are not consistent
                if (visit1->second != visit2->second)
                {
                    xs_visited.erase(visit1);
                    xs_visited.erase(visit2);
                }
                continue; // Skip if two observations are already visited
            }

            uint X_idx = 0;
            if (visit1 != xs_visited.end()) X_idx = visit1->second;
            else if (visit2 != xs_visited.end()) X_idx = visit2->second;
            else
            {
                // Add a new point if two observations are not visited
                X_idx = uint(Xs.size());
                Xs.push_back(cv::Point3d(0, 0, Z_init));
                Xs_rgb.push_back(img_set[cam1_idx].at<cv::Vec3b>(Image_info_[cam1_idx].features[x1_idx].pt));
            }
            if (visit1 == xs_visited.end()) xs_visited[key1] = X_idx;
            if (visit2 == xs_visited.end()) xs_visited[key2] = X_idx;
        }
    }
    printf("# of 3D points: %zd\n", Xs.size());

    // 3) Optimize camera pose and 3D points together (bundle adjustment)
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int cam_idx = SFM::getCamIdx(visit->first), x_idx = SFM::getObsIdx(visit->first);
        const cv::Point2d& x = Image_info_[cam_idx].features[x_idx].pt;
        SFM::addCostFunc6DOF(ba, Xs[visit->second], x, cameras[cam_idx], ba_loss_width);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    if (ba_num_iter > 0) options.max_num_iterations = ba_num_iter;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::vector<std::vector<cv::KeyPoint>> img_keypoint;
    img_keypoint.reserve(Image_info_.size());
    for (size_t mm = 0; mm < Image_info_.size(); mm++) {
        img_keypoint.emplace_back(Image_info_[mm].features);
    }

    // Mark erroneous points to reject them
    std::vector<bool> is_noisy = maskNoisyPoints(Xs, img_keypoint, cameras, xs_visited, ba_loss_width);
    int num_noisy = std::accumulate(is_noisy.begin(), is_noisy.end(), 0);

    assert(Image_info_.size == cameras.size());

    for (size_t j = 0; j < cameras.size(); j++) {

        double min_z, max_z;
        cv::Vec3d rvec(cameras[j][0], cameras[j][1], cameras[j][2]), t(cameras[j][3], cameras[j][4], cameras[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;

        cv::Vec3d cam2pt(Xs[0].x - p[0], Xs[0].y - p[1], Xs[0].z - p[2]);
        cv::Vec3d normal(R.t()(0, 2), R.t()(1, 2), R.t()(2, 2));


        // compute minimum and maximum of depth
        double z = cam2pt.dot(normal);
        min_z = z;
        max_z = z;
        for (size_t i = 1; i < Xs.size(); i++)
        {
            if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit && !is_noisy[i]) {
                cv::Vec3d cam2pt(Xs[i].x - p[0], Xs[i].y - p[1], Xs[i].z - p[2]);
                cv::Vec3d normal(R.t()(0, 2), R.t()(1, 2), R.t()(2, 2));
                double z_temp = cam2pt.dot(normal);
                if (z_temp > 0) {
                    if (z_temp < min_z) { min_z = z_temp; }
                    else if (z_temp > max_z) { max_z = z_temp; }
                }
            }
        }

        Image_info_[j].max_z_val = max_z;
        Image_info_[j].min_z_val = min_z;

        RotationMatrixToEulerAngles(R, Image_info_[j].yaw, Image_info_[j].pitch, Image_info_[j].roll);
        TranslationToPosition(p, 1, Image_info_[j].pos0, Image_info_[j].pos1, Image_info_[j].pos2);
        Image_info_[j].focal = cameras[j][6];
        Image_info_[j].c_x = cameras[j][7];
        Image_info_[j].c_y = cameras[j][8];

        fprintf(stdout, "Camera %zd's position (axis_0, axis_1, axis_2) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].pos0, Image_info_[j].pos1, Image_info_[j].pos2);
        fprintf(stdout, "Camera %zd's rotation (yaw, pitch, roll) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].yaw, Image_info_[j].pitch, Image_info_[j].roll);
        fprintf(stdout, "Camera %zd's (f, cx, cy) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].focal, Image_info_[j].c_x, Image_info_[j].c_y);
        fprintf(stdout, "Camera %zd's z range  = [%.3f (=min) , %.3f (=max)]\n\n", j, min_z, max_z);

    }

    std::cout << "# of image " << Image_info_.size() << std::endl;
 
    return true;
}

bool Write_JSON_file(const std::vector<Img_info>& Image_info_, const char* result_path) {
    if (Image_info_.empty()) return false;
    int name_start_pos = Image_info_[0].FileName.find("\\", 0) > Image_info_[0].FileName.find("/", 0) ? Image_info_[0].FileName.find("\\", 0) + 1 : Image_info_[0].FileName.find("/", 0) + 1;
    FILE* fpts;
    fopen_s(&fpts, result_path, "w");
    fprintf(fpts, "{\n");
    fprintf(fpts, "  \"cameras\":[\n");
    for (int pic_idx = 0; pic_idx < Image_info_.size(); pic_idx++) {

        std::string zero_idx;
        int num_img = Image_info_.size();
        if (num_img < 100 && pic_idx < 10) zero_idx = "0";
        else if (num_img > 100 && pic_idx < 10) zero_idx = "00";
        else if (num_img > 100 && pic_idx < 100) zero_idx = "0";
        else zero_idx = "";
        std::string view_name = "v" + zero_idx + std::to_string(Image_info_[pic_idx].img_idx);

        fprintf(fpts, "    {\n");
        fprintf(fpts, "      \"Depthmap\": 1,\n");
        fprintf(fpts, "      \"Background\": 0,\n");
        fprintf(fpts, "      \"BitDepthColor\": 8,\n");
        fprintf(fpts, "      \"BitDepthDepth\": 16,\n");
        fprintf(fpts, "      \"ColorSpace\": \"YUV420\",\n");
        fprintf(fpts, "      \"DepthColorSpace\": \"YUV420\",\n");
        fprintf(fpts, "      \"Projection\": \"Perspective\",\n");
        fprintf(fpts, "      \"Name\": \"%s\",\n", view_name);
        fprintf(fpts, "      \"Position\": [%.6f, %.6f, %.6f],\n", Image_info_[pic_idx].pos0, Image_info_[pic_idx].pos1, Image_info_[pic_idx].pos2);
        fprintf(fpts, "      \"Rotation\": [%.6f, %.6f, %.6f],\n", Image_info_[pic_idx].yaw, Image_info_[pic_idx].pitch, Image_info_[pic_idx].roll);
        fprintf(fpts, "      \"Focal\": [%.6f, %.6f],\n", Image_info_[pic_idx].focal, Image_info_[pic_idx].focal);
        fprintf(fpts, "      \"Principle_point\": [%.6f, %.6f],\n", Image_info_[pic_idx].c_x, Image_info_[pic_idx].c_y);
        fprintf(fpts, "      \"Depth_range\": [%.6f, %.6f],\n", Image_info_[pic_idx].min_z_val, Image_info_[pic_idx].max_z_val);
        fprintf(fpts, "      \"Resolution\": [%d, %d]\n", Image_info_[pic_idx].Image.cols, Image_info_[pic_idx].Image.rows);
        if(pic_idx != Image_info_.size()-1) fprintf(fpts, "    },\n");
        else fprintf(fpts, "    }\n");
    }

    fprintf(fpts, "  ]\n");
    fprintf(fpts, "}\n");
}


int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stdout, "ERROR::No input dataset\nIn order to run the program type: 'Pose_Estimation.exe [the path of input data] [the name of output json file] [(optional)scale factor] [(optional)the path of output picture] [(optional)data name]'\n");
        fprintf(stdout, "EXAMPLE::Pose_Estimation.exe ./pic cam_param.json 1 ./out_pic GB_U'\n");
        return -1;
    }
    
    cv::String path_ = argv[1];
    const char* result_text_ = argv[2];
    float scale_f = -1.0;
    cv::String out_path_ = argv[1];
    cv::String data_name = "scaled";

    if (argc > 3) scale_f = std::stof(argv[3]);
    if (argc > 4) out_path_ = argv[4];
    if (argc > 5) data_name = argv[5];

    if (scale_f < epsilon_) scale_f = 1.0;

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

    if (num_images >= 1000) {
        std::cerr << "Too many pictures to process" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> org_img_set(num_images);
    std::vector<cv::Mat> scaled_img_set(num_images);
    std::vector<std::string> scaled_img_names(num_images);
    std::cout << "# of pictures : " << num_images << std::endl;
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

        resize(org_img_set[i], scaled_img_set[i], cv::Size(), scale_f, scale_f);
        std::string zero_idx;
        if (num_images < 100 && i < 10) zero_idx = "0";
        else if (num_images > 100 && i < 10) zero_idx = "00";
        else if (num_images > 100 && i < 100) zero_idx = "0";
        else zero_idx = "";
        scaled_img_names[i] = std::to_string(scaled_img_set[i].cols) + "x" + std::to_string(scaled_img_set[i].rows) + "_" + zero_idx + std::to_string(i) +  ".png";
        scaled_img_names[i] = data_name + "_" + scaled_img_names[i];
        scaled_img_names[i] = out_path_ + "/" + scaled_img_names[i];    
    }
    std::vector<Img_info> Image_info;
    if (!pose_estimatoin(scaled_img_set, scaled_img_names, Image_info)) {
        std::cerr << "[ERROR::Pose_Estimation]\n";
    }

    for (int img_idx = 0; img_idx < Image_info.size(); img_idx++) {
        cv::imwrite(scaled_img_names[img_idx], scaled_img_set[img_idx]);
    }

    if (!Write_JSON_file(Image_info, result_text_)) {
        std::cerr << "[ERROR::Write_JSON_file]\n";
    }
    std::cout << "[DONE]" << std::endl;
    return 0;
}
