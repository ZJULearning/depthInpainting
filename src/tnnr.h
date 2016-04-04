// C++ reimplementation of TNNR Matlab Code by Debing Zhang
// Hongyang Xue
// 2016-01-31
#ifndef __TNNR_H_
#define __TNNR_H_
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
Mat APGL(Mat &A, Mat &B, Mat &X, Mat &M, Mat &mask, float eps, float lambda);
Mat TNNR(Mat &im0, Mat &mask, int lower_R, int upper_R, float lambda = 0.06);
#endif
