/************************************************
* Copyright 2019 GWM India R&D Pvt Ltd. All Rights Reserved.
* File: skyline_extract.cc
* Date: 24-May-2019 *Version: 1.0 - Initial version
* Date: 17-Jul-2019 *Version: 1.1 - Modified for improved detection
************************************************/

#include <iostream>
#include <sky_extract.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

#define SKY_WIDTH 1920
#define SKY_HEIGHT 1440

int low_H = 0, low_S = 0, low_V = 140;
int high_H = 200, high_S = 120, high_V = 255;
RNG rng(12345);

int main(int argc, char** argv)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    char image_file_path[100];  /*path of the image folder*/
    int image_num = 0;          /*image number to increment*/

    if(argc < 1)
    {
    	cout << "Please enter required arguements" << endl;
        getchar();
    }

    while(1)
    {
        sprintf(image_file_path, "./upview/left%04d.jpg",image_num);
        
	// Load an image	
	skyDetector skydet_obj;
	if(!skydet_obj.load_image(image_file_path))
	{
	     cout << "/nImage not valid or end of image sequence" << endl;
	     getchar();
             return 0;
	}

	//Select RoI - Load only 1200 x 800 image
	Rect roi = Rect(0, 0, SKY_WIDTH, SKY_HEIGHT);

	// Convert the image to hsv
	cv::Mat src_hsv = skydet_obj.RGB2HSV(skydet_obj.src_img, src_hsv);
	src_hsv = src_hsv(roi);
        namedWindow("Souce_HSV", 0);
	cv::resizeWindow("Souce_HSV",Size(640,480));
	imshow("Souce_HSV", src_hsv);
	moveWindow("Souce_HSV", 0,0);

	// set high and low ranges for Hue, Sat & Value to detect sky region
	cv::Mat sky_area = Mat::zeros(src_hsv.size(), CV_8UC3);
	sky_area = skydet_obj.HSV_Range(src_hsv, sky_area);
	namedWindow( "sky_area", 0 );
	cv::resizeWindow("sky_area",Size(640,480));
	imshow("sky_area", sky_area);
	moveWindow("sky_area", 700,0);

	//find sky area contour
	findContours(sky_area, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	//Draw the contour
	Mat drawing = Mat::zeros(skydet_obj.src_img.size(), CV_8UC3);
	for(int i = 0; i< contours.size(); i++)
	{
	    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}
	
	// Show in a window
	namedWindow( "sky_contour", 0 );
	cv::resizeWindow("sky_contour",Size(640,480));
	imshow("sky_contour", drawing);
	moveWindow("sky_contour", 1280,0);

	vector< vector<Point> > hull(contours.size());
	for(int i = 0; i < contours.size(); i++)
	{
	    convexHull(Mat(contours[i]), hull[i], false);
	}

	// create a blank image (black image)
	Mat drawing_cnvx = Mat::zeros(skydet_obj.src_img.size(), CV_8UC3);

	for(int i = 0; i < contours.size(); i++)
	{
	    Scalar color_contours = Scalar(0, 255, 0); // green - color for contours
	    Scalar color = Scalar(255, 0, 150); // blue - color for convex hull
	    // draw ith contour
	    drawContours(drawing_cnvx, contours, i, color_contours, 1, 8, vector<Vec4i>(), 0, Point(skydet_obj.src_img.cols/4,0));
	    // draw ith convex hull
	    drawContours(drawing_cnvx, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point(skydet_obj.src_img.cols/4,0));
	}

	// Show in a window
        namedWindow( "Convex_hull", 0 );
	cv::resizeWindow("Convex_hull",Size(640,480));
	imshow( "Convex_hull", drawing_cnvx );
	moveWindow("Convex_hull", 0,581);


	/*find out approximate polygon with accuracy +/-3*/
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>centers( contours.size() );
	vector<float>radius( contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
	{
	    approxPolyDP( contours[i], contours_poly[i], 3, true );
	    boundRect[i] = boundingRect( contours_poly[i] );
	    minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
	}

        Mat BB = Mat::zeros(skydet_obj.src_img.size(), CV_8UC3 );
        for( size_t i = 0; i< contours.size(); i++ )
	{
	   Scalar color = Scalar( 0,0,250 );
	   drawContours(skydet_obj.src_img, contours_poly, (int)i, color, -1 );
	   if(boundRect[i].height > 100 && boundRect[i].width > 100)
	   {	     	
              // putText(skydet_obj.src_img,"sky region",Point(boundRect[i].height/2, boundRect[i].width/2),3,2,Scalar(250,250,255),3);
	   }
        }
        
        // Show in a window 
   	namedWindow( "boundBox", 0 );
   	cv::resizeWindow("boundBox",640, 480);
    	imshow("boundBox", skydet_obj.src_img);
        moveWindow("boundBox", 750,580);
	if(waitKey(30) == 27) // Wait for 'esc' key press to exit
    	{
            break;
    	}
    	image_num++;
    }
    return 0;

}

bool skyDetector::load_image(const std::string &image_file_path)
{
	src_img = imread(image_file_path, CV_LOAD_IMAGE_UNCHANGED);
	if(src_img.empty())
	{
	    cout << "image Load error";
	    return false;
	}
	return true;
}

cv::Mat skyDetector::RGB2HSV(const cv::Mat &src_img, cv::Mat &src_hsv)
{
	cv::Mat hsv = Mat::zeros(src_img.size(), CV_8UC3);
	cvtColor( src_img, hsv, CV_BGR2HSV );
	return hsv;
}

cv::Mat skyDetector::HSV_Range(const cv::Mat &src_hsv, cv::Mat &dst_hsv)
{
	inRange(src_hsv, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), dst_hsv);
	return dst_hsv;
}

