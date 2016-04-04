#ifndef __UTIL_H_
#define __UTIL_H_
#include <opencv2/opencv.hpp>
using namespace cv;

float PSNR(Mat &original, Mat &inpainted, Mat &mask);

float NORM_TV(Mat &img);
#endif
