#ifndef __LRTV_H__
#define __LRTV_H__

#include <opencv2/opencv.hpp>
#include <iostream>
#include "tnnr.h"
using namespace std;
using namespace cv;

// Low-Rank Total Variation for Image Inpainting
class LRTV
{
 protected:
  
  Mat U_;
  Mat U_last_;
  Mat mask_; //mask for the missing pixels: 0 indicates missing
  Mat I_, M_, Y_;
  int W_, H_;
  float rho_, dt_, h_, lambda_tv_;
  float lambda_rank_, alpha_;
 
  Mat kernelx_plus_, kernelx_minus_;
  Mat kernely_plus_, kernely_minus_;

 public:
  LRTV(){};
  LRTV(Mat &I, Mat &mask);
  // debug purpose
  void init_U(Mat &u0);
  Mat getU();
  Mat getY();
  Mat getM();
  //
  void setParameters(float rho, float dt, float lambda_tv, float lambda_rank);
  float sub_1_val(Mat &X); // evaluate subproblem 1 to guide gradient descent
  void sub_1();
  void sub_2();
  void sub_3();
  Mat compute();
};

#endif
