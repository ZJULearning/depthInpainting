#ifndef __LRTVPHI_H__
#define __LRTVPHI_H__

// Transformed(Shrinkage) TV Norm Regularization
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tnnr.h"
#include "LRTV.h"
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SparseQR>
using namespace std;
using namespace cv;

// Low-Rank Shrinked Total Variation for Depth Image Inpainting
class LRTVPHI:public LRTV
{
  inline int ind21(int i, int j);
 protected:
  float a_, b_; // parameters for function transform
  float gamma_;
  float epsilon_d_;
  float transform(float x);
  float dtrans(float x);
  float d2trans(float x);
  float transform_h(float x);
  float dtrans_h(float x);
  float d2trans_h(float x);
  Mat px_, py_, m_;
  Mat px_tilde, py_tilde;
  Mat U_x_, U_y_;
  Mat inverse_phi_m_;
  Mat phi_m_;
  Mat chi_;
  Mat d_phi_m_over_phi_m_;

  Mat AX, AY, BX, BY, DX, DY;
  Mat hessian_Theta;
  float *_DX;
  float *_DY;
  float *_mask;
 public:
  LRTVPHI(){};
  LRTVPHI(Mat &I, Mat &mask);
  void setShrinkageParam(float a, float b);
  void setNewtonParameters(float gamma);
  void init_M(Mat &X);
  float sub_1_val(Mat &X); // evaluate subproblem 1 to guide gradient descent
  void sub_1(float, float);
  Mat sub_1_grad(Mat &X);
  Mat compute();
  float NORM_TVPHI(Mat &X);
  float NORM_TVPHI_h(Mat &X);
  //

  
  Mat laplacian_D(float alpha, float epsilon, Mat &DX, Mat &DY, Mat &d);
  Mat make_g();
  Mat solveForDirection(float, float, Mat &, bool &);
  void make_H_R(float);
  //
};

inline int LRTVPHI::ind21(int i, int j)
{
  return i * W_ + j;
}
#endif
