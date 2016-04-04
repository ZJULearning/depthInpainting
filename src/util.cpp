#include "util.h"

//  PSNR: written on 16-01-27. Testing Passed!
float PSNR(Mat &Xfull_, Mat &Xrecover_, Mat &mask_)
{
  Mat Xfull, Xrecover, mask;
  Xfull_.convertTo(Xfull, CV_32F);
  Xrecover_.convertTo(Xrecover, CV_32F);
  mask_.convertTo(mask, CV_32F);

  max(0.0, Xrecover, Xrecover);
  min(255.0, Xrecover, Xrecover);

  float MSE = 0.0;
  Mat missing = (255 - mask) / 255;
  Mat diff = Xfull - Xrecover;
  Mat result;
  multiply(diff, missing, result);

  // NORM_L2: frobenius norm
  MSE = norm(result, NORM_L2);
  MSE = MSE * MSE;
  int nnz = countNonZero(missing);

  MSE = MSE / static_cast<float>(nnz);

  float psnr = 10 * log10(255 * 255 / MSE);
  return psnr;
}

// NORM_TV
float NORM_TV(Mat &img)
{
  Mat U;
  img.convertTo(U, CV_32FC1);
  // make gradient operators
  Mat kernelx_plus_ = (Mat_<float>(1,3)<<0.0,-1.0,1.0);
  Mat kernelx_minus_ = (Mat_<float>(1,3)<<-1.0,1.0,0.0);
  Mat kernely_plus_ = (Mat_<float>(3,1)<<0.0,-1.0,1.0);
  Mat kernely_minus_ = (Mat_<float>(3,1)<<-1.0,1.0,0.0);

  Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
  // default border type: BORDER_REFLECT101
  filter2D(U, grad_x_plus, -1, kernelx_plus_);
  filter2D(U, grad_x_minus, -1, kernelx_minus_);
  filter2D(U, grad_y_plus, -1, kernely_plus_);
  filter2D(U, grad_y_minus, -1, kernely_minus_);

  Mat U_x = (grad_x_minus + grad_x_plus) / 2.0;
  Mat U_y = (grad_y_minus + grad_y_plus) / 2.0;
  pow(U_x, 2, U_x);
  pow(U_y, 2, U_y);

  Mat grad;
  sqrt(U_x + U_y, grad);
  return sum(grad)[0];
}
