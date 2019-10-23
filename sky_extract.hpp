/*
 * skyline_classic.hpp
 *
 *  Created on: 22-May-2019
 *      
 */

#ifndef SKYLINE_CLASSIC_HPP_
#define SKYLINE_CLASSIC_HPP_

#include <string>
#include <opencv2/opencv.hpp>

class skyDetector{

public:
	skyDetector() = default;
	~skyDetector() = default;

	cv::Mat src_img;

	bool load_image(const std::string &image_file_path);
	cv::Mat RGB2HSV(const cv::Mat &src_img, cv::Mat &src_hsv);
        cv::Mat HSV_Range(const cv::Mat &src_hsv, cv::Mat &dst_hsv);
};


#endif /* SKYLINE_CLASSIC_HPP_ */

