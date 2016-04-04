#include "LRTVPHI.h"
//#define _DEBUG_

#define _HOLY_
void LRTVPHI::setNewtonParameters(float gamma)
{
  gamma_ = gamma;
}

LRTVPHI::LRTVPHI(Mat &I, Mat &mask):LRTV(I, mask)
{
  
}

void LRTVPHI::setShrinkageParam(float a, float b)
{
  a_ = a;
  b_ = b;
}

float LRTVPHI::transform_h(float x)
{
  if (abs(x) >= gamma_)
    return transform(x) - transform(gamma_) + gamma_ * dtrans(gamma_) / 2.0f;
  else
    return dtrans(gamma_) * pow(x,2) / (2.0 * gamma_);
}

float LRTVPHI::transform(float x)
{
  #ifdef _HOLY_
  return sqrt(abs(x));
  #else
  return a_*tanh(b_*x);
  #endif
}

float LRTVPHI::dtrans(float x)
{
  #ifdef _HOLY_
  return 0.5 / sqrt(abs(x));
#else  
  return a_*b_*(1.0 - pow(tanh(b_*x),2.0));
  #endif
}


float LRTVPHI::d2trans(float x)
{
  #ifdef _HOLY_
  return -0.25*pow(x, -1.5);
  #else
  float t = tanh(b_*x);
  return -2.0*a_*b_*t*(1-t*t);
  #endif
}

float LRTVPHI::dtrans_h(float x)
{
  if(abs(x) >= gamma_)
    return dtrans(x);
  else
    return dtrans(gamma_) / gamma_ * x;
 }

float LRTVPHI::d2trans_h(float x)
{
  if(abs(x) >= gamma_)
    return d2trans(x);
  else
    return dtrans(gamma_) / gamma_;
 
}

// 2016-02-06
float LRTVPHI::NORM_TVPHI(Mat &img)
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

  for(int i = 0; i < grad.rows; ++i)
    for(int j = 0; j < grad.cols; ++j)
      grad.at<float>(i,j) = transform(grad.at<float>(i,j));

  // apply transform to every element of grad
  // Mat exp1, exp0;
  // exp(grad,exp1);
  // exp(-grad, exp0);
  // divide(exp1 - exp0, exp1 + exp0, grad);

 
  return sum(grad)[0];
}

float LRTVPHI::NORM_TVPHI_h(Mat &img)
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

  for(int i = 0; i < grad.rows; ++i)
    for(int j = 0; j < grad.cols; ++j)
      grad.at<float>(i,j) = transform_h(grad.at<float>(i,j));

  return sum(grad)[0];
}

void LRTVPHI::init_M(Mat &X)
{
  M_ = X.clone();
  M_.convertTo(M_, CV_32FC1);
}
// objective function
float LRTVPHI::sub_1_val(Mat &X)
{
  float val = 0.0;
  Mat part1 = (X - I_);
  multiply(part1, mask_, part1);
  pow(part1, 2.0, part1);

  val += sum(part1)[0];

  val += lambda_tv_ * NORM_TVPHI_h(X);

  Mat part2 = X - M_ + Y_;
  pow(part2, 2.0, part2);
  part2 = rho_ / 2.0 * part2;

  val += sum(part2)[0];

  cout << "TV part = " << lambda_tv_  * NORM_TVPHI_h(X) << endl;
  cout << "image part = " << val - lambda_tv_ * NORM_TVPHI_h(X) << endl;
  return val;
}

Mat LRTVPHI::sub_1_grad(Mat &X)
{
  float epsilon = 1.0e-5;
  Mat part1 = 2.0  * (X - I_);
  multiply(part1, mask_, part1);
  part1 = part1 + rho_ * (X - M_ + Y_);

  //

  Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
  // default border type: BORDER_REFLECT101
  filter2D(X, grad_x_plus, -1, kernelx_plus_);
  filter2D(X, grad_x_minus, -1, kernelx_minus_);
  filter2D(X, grad_y_plus, -1, kernely_plus_);
  filter2D(X, grad_y_minus, -1, kernely_minus_);

  Mat U_xx, U_yy, U_x, U_y, U_xy;
  filter2D(grad_x_minus, U_xx, -1, kernelx_plus_);
  filter2D(grad_y_minus, U_yy, -1, kernely_plus_);

  U_x = (grad_x_minus + grad_x_plus) / 2.0;
  U_y = (grad_y_minus + grad_y_plus) / 2.0;

  Mat U_y_x_plus, U_y_x_minus;
  filter2D(U_y, U_y_x_plus, -1, kernelx_plus_);
  filter2D(U_y, U_y_x_minus, -1, kernelx_minus_);
  U_xy = (U_y_x_plus + U_y_x_minus) / 2.0;

  // update U_
  // element-wise multiplication results
  Mat uy_uy, ux_uy, ux_ux, uxy_ux_uy, uxx_uy_uy, uyy_ux_ux;
  Mat numerator;
  multiply(U_y, U_y, uy_uy);
  uy_uy = uy_uy + epsilon;
  multiply(U_x, U_x, ux_ux);
  ux_ux = ux_ux + epsilon;
  multiply(U_x, U_y, ux_uy);

  multiply(U_xx, uy_uy, uxx_uy_uy);
  multiply(U_yy, ux_ux, uyy_ux_ux);
  multiply(U_xy, ux_uy, uxy_ux_uy);
  numerator = uxx_uy_uy - 2.0 * uxy_ux_uy + uyy_ux_ux;

  Mat denominator = ux_ux + uy_uy - epsilon;

  Mat denominator1_5;
  pow(denominator, 1.5, denominator1_5);

  Mat part2;
  Mat part2_1, part2_2;
  divide(numerator, denominator1_5, part2_1);

  Mat uxx_ux_ux, uyy_uy_uy;
  multiply(U_xx, ux_ux, uxx_ux_ux);
  multiply(U_yy, uy_uy, uyy_uy_uy);
  Mat numerator_1 = uxx_ux_ux + 2.0 * uxy_ux_uy + uyy_uy_uy;
  numerator_1 = numerator_1 - epsilon * (U_xx + U_yy);
  divide(numerator_1, denominator, part2_2);
  Mat dtransM = Mat::zeros(H_, W_, CV_32FC1);
  Mat d2transM = Mat::zeros(H_, W_, CV_32FC1);

  Mat mod;
  sqrt(denominator, mod);

  float *x = reinterpret_cast<float*>(mod.data);
  float *d1 = reinterpret_cast<float*>(dtransM.data);
  float *d2 = reinterpret_cast<float*>(d2transM.data);
  for(int i = 0; i < H_ * W_; ++i)
    {
      d1[i] = dtrans_h(x[i]);
      d2[i] = d2trans_h(x[i]);
    }

  multiply(part2_1, dtransM, part2_1);
  multiply(part2_2, d2transM, part2_2);
  part2 = part2_1 + part2_2;
  part2 = lambda_tv_ * part2;     
  return part1 + part2;
}


Mat LRTVPHI::make_g()
{
  // mask_: 0-1 matrix
  Mat g = 2 * (U_ - I_);
  multiply(g, mask_, g);
  g = g + rho_ * (U_ - M_ + Y_);


  Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
  // default border type: BORDER_REFLECT101
  filter2D(U_, grad_x_plus, -1, kernelx_plus_);
  filter2D(U_, grad_x_minus, -1, kernelx_minus_);
  filter2D(U_, grad_y_plus, -1, kernely_plus_);
  filter2D(U_, grad_y_minus, -1, kernely_minus_);

  Mat U_x, U_y;
  
  U_x = (grad_x_minus + grad_x_plus) / 2.0;
  U_y = (grad_y_minus + grad_y_plus) / 2.0;

  U_x.copyTo(U_x_);
  U_y.copyTo(U_y_);
  Mat U_x2, U_y2;
  // make m
  multiply(U_x, U_x, U_x2);
  multiply(U_y, U_y, U_y2);

  
  Mat gradAbs = U_x2 + U_y2;
  sqrt(gradAbs, gradAbs);

  max(gradAbs, gamma_, m_);

  // characteristic matrix 
  threshold(gradAbs, chi_, gamma_, 1.0, THRESH_BINARY);
  m_.copyTo(inverse_phi_m_);
  m_.copyTo(phi_m_);
  float *inverse_phi_m_data = reinterpret_cast<float*>(inverse_phi_m_.data);
  float *phi_m_data = reinterpret_cast<float*>(phi_m_.data);
  for(int i = 0; i < W_ * H_; ++i)
    {
      inverse_phi_m_data[i] = dtrans(inverse_phi_m_data[i]) / inverse_phi_m_data[i];
      phi_m_data[i] = (phi_m_data[i]) / dtrans(phi_m_data[i]);
    }

  //cout << phi_m_ << endl; exit(0);
  Mat tempX, tempY;
  multiply(inverse_phi_m_, U_x, tempX);
  multiply(inverse_phi_m_, U_y, tempY);

  Mat U_xx, U_yy;
  filter2D(tempX, U_xx, -1, kernelx_plus_);
  filter2D(tempY, U_yy, -1, kernely_plus_);

  g = g - lambda_tv_ * (U_xx + U_yy);

  return g;
}

// the problem is H!
// H can be very very large...
// Luckily it is sparse
// We need to construct a sparse large matrix H
// 2016-02-26

void LRTVPHI::make_H_R(float epsilon)
{
  hessian_Theta = Mat::ones(H_, W_, CV_32FC1);
  hessian_Theta = 2.0 * hessian_Theta;
  multiply(hessian_Theta, mask_, hessian_Theta);
  hessian_Theta = hessian_Theta + rho_;

  // diagonal matrix in H
  // make \tilde{p^k}
  Mat temp1, up1, downx1, downy1, upx1, upy1;
  multiply(m_, inverse_phi_m_, temp1);
  multiply(chi_, temp1, up1);

  multiply(up1, px_, upx1);

  multiply(up1, py_, upy1);
  max(temp1, abs(px_), downx1);
  max(temp1, abs(py_), downy1);
  // Mat px_tilde, py_tilde;
  divide(upx1, downx1, px_tilde);
  divide(upy1, downy1, py_tilde);


  Mat d_phi_m;
  m_.copyTo(d_phi_m_over_phi_m_);
  float *d_phi_m_over_phi_m_data = reinterpret_cast<float*>(d_phi_m_over_phi_m_.data);
  for(int i = 0; i < W_ * H_ ; ++i)
    {
      float m = d_phi_m_over_phi_m_data[i];
      d_phi_m_over_phi_m_data[i] = 1.0 / m + 2 * b_ * transform(b_ * m);
    }

  Mat tempX, tempY;
  multiply(d_phi_m_over_phi_m_, U_x_, tempX);
  multiply(d_phi_m_over_phi_m_, U_y_, tempY);
    
  multiply(tempX, px_tilde, tempX);
  multiply(tempY, py_tilde, tempY);
  divide(tempX, m_, tempX);
  divide(tempY, m_, tempY);
  AX = inverse_phi_m_ - tempX;
  AY = inverse_phi_m_ - tempY;

  // up here we have made the diagonal matrix in H
  
  // for \tilde{R}^k

  // Mat BX, BY;
  multiply(d_phi_m_over_phi_m_, U_x_, tempX);
  multiply(d_phi_m_over_phi_m_, U_y_, tempY);

  divide(tempX, m_, tempX);
  divide(tempY, m_, tempY);
  
  
  multiply(px_tilde, tempX, BX);
  multiply(py_tilde, tempY, BY);
  
}

Mat LRTVPHI::laplacian_D(float alpha, float epsilon, Mat &DX, Mat &DY, Mat &d)
{
  Mat result = Mat::zeros(H_, W_, CV_32FC1);
  float *_DX = reinterpret_cast<float*>(DX.data);
  float *_DY = reinterpret_cast<float*>(DY.data);
  float *_result = reinterpret_cast<float*>(result.data);
  float *_d = reinterpret_cast<float*>(d.data);
  for(int i = 0; i < H_; ++i)
    {
      int i_minus_1 = i - 1;
      int i_plus_1 = i + 1;
      int i_minus_2 = i - 2;
      int i_plus_2 = i + 2;

      if(i_minus_1 < 0) i_minus_1 = 1;
      if(i_minus_2 < 0) i_minus_2 = 2 - i;
      if(i_plus_1 > H_ - 1) i_plus_1 = H_  - 2;
      if(i_plus_2 > H_ - 1) i_plus_2 = 2*H_ - 4 - i_plus_2;
      for(int j = 0; j < W_; ++j)
	{
	  int j_minus_1 = i - 1;
	  int j_plus_1 = j + 1;
	  int j_minus_2 = j - 2;
	  int j_plus_2 = j + 2;
	  if(j_minus_1 < 0) j_minus_1 = 1;
	  if(j_minus_2 < 0) j_minus_2 = 2 - j;
	  if(j_plus_1 > W_ - 1) j_plus_1 = W_ - 2;
	  if(j_plus_2 > W_ - 1) j_plus_2 = 2*W_ - 4 - j_plus_2;

	  int ind = ind21(i, j);
	  _result[ind] = _DX[ind21(i, j_plus_1)] * (_d[ind21(i, j_plus_2)] - _d[ind]) - \
	    _DX[ind21(i, j_minus_1)] * (_d[ind] - _d[ind21(i, j_minus_2)]) + \
	    _DY[ind21(i_plus_1, j)] * (_d[ind21(i_plus_2, j)] - _d[ind]) - \
	    _DY[ind21(i_minus_1, j)] * (_d[ind] - _d[ind21(i_minus_2, j)]);
	}
    }

  result = -alpha / 4.0 * result;
  return result;
}

Mat LRTVPHI::solveForDirection(float epsilon, float beta, Mat &g, bool &sucess)
{
  epsilon_d_ = 1e-8;
  cout << "epsilon_d_ = " << epsilon_d_ << endl;
  DX = AX + beta * BX;
  DY = AY + beta * BY;
  _DX = reinterpret_cast<float*>(DX.data);
  _DY = reinterpret_cast<float*>(DY.data);
  _mask = reinterpret_cast<float*>(mask_.data);
  //convert to vector form and solve the linear system

  Eigen::SparseMatrix<float> M(H_*W_, H_*W_);
  M.reserve(Eigen::VectorXi::Constant(H_*W_, 7));

  Eigen::VectorXf vg, du;

  // construct sparseMatrix M

  // cout << AX << endl;
  // exit(0);
  // float epsilon = 1.0e-6;	
  for(int i = 0; i < H_; ++i)
    {
      int i_minus_1 = i - 1;
      int i_plus_1 = i + 1;
      int i_minus_2 = i - 2;
      int i_plus_2 = i + 2;

      if(i_minus_1 < 0) i_minus_1 = 1;
      if(i_minus_2 < 0) i_minus_2 = 2 - i;
      if(i_plus_1 > H_ - 1) i_plus_1 = H_  - 2;
      if(i_plus_2 > H_ - 1) i_plus_2 = 2*H_ - 4 - i_plus_2;
      for(int j = 0; j < W_; ++j)
	{
	  int j_minus_1 = i - 1;
	  int j_plus_1 = j + 1;
	  int j_minus_2 = j - 2;
	  int j_plus_2 = j + 2;
	  if(j_minus_1 < 0) j_minus_1 = 1;
	  if(j_minus_2 < 0) j_minus_2 = 2 - j;
	  if(j_plus_1 > W_ - 1) j_plus_1 = W_ - 2;
	  if(j_plus_2 > W_ - 1) j_plus_2 = 2*W_ - 4 - j_plus_2;

	  int k = i * W_ + j;
	  int l;

	  
	  l = ind21(i, j);
	  M.coeffRef(k, l) += hessian_Theta.at<float>(i,j) + epsilon * beta + lambda_tv_ / 4.0 * (_DX[ind21(i, j_plus_1)] + _DX[ind21(i, j_minus_1)] + _DX[ind21(i_plus_1, j)] + _DX[ind21(i_minus_1, j)]);

	  l = ind21(i, j_plus_2);
	  M.coeffRef(k, l) += -lambda_tv_ / 4.0 * _DX[ind21(i, j_plus_1)];

	  l = ind21(i, j_minus_2);
	  M.coeffRef(k, l) += -lambda_tv_ / 4.0 * _DX[ind21(i, j_minus_1)];

	  l = ind21(i_plus_2, j);
	  M.coeffRef(k, l) += -lambda_tv_ / 4.0 * _DY[ind21(i_plus_1, j)];

	  l = ind21(i_minus_2, j);
	  M.coeffRef(k, l) += -lambda_tv_ / 4.0 * _DY[ind21(i_minus_1, j)];

	  // hahaha
	  
	  // l = ind21(i, j);
	  // M.coeffRef(k, l) += hessian_Theta.at<float>(i, j) + epsilon * beta + lambda_tv_ / 2.0 * (_DX[ind21(i, j_plus_1)] + _DY[(ind21(i_plus_1, j))]);

	  // l = ind21(i, j_plus_2);
	  // M.coeffRef(k, l) += -lambda_tv_ / 2.0 * _DX[ind21(i, j_plus_1)];

	  // l = ind21(i, j_plus_1);
	  // M.coeffRef(k, l) += lambda_tv_ / 2.0 * _DX[ind21(i,j)];

	  // l = ind21(i, j_minus_1);
	  // M.coeffRef(k, l) += -lambda_tv_ / 2.0 * _DX[ind21(i,j)];

	  // l = ind21(i_plus_2, j);
	  // M.coeffRef(k, l) += -lambda_tv_ / 2.0 * _DY[ind21(i_plus_1, j)];

	  // l = ind21(i_plus_1, j);
	  // M.coeffRef(k, l) += lambda_tv_ / 2.0 * _DY[ind21(i_plus_1, j)];

	  // l = ind21(i_minus_1, j);
	  // M.coeffRef(k, l) += -lambda_tv_ / 2.0 * _DY[ind21(i,j)];
	}
    }

  cout << "   to make g" << endl;
  // make -g
  float *g_data = reinterpret_cast<float*>(g.data);
  vg.resize(H_*W_);
  for(int i = 0; i < H_*W_; ++i)
    vg(i) = -g_data[i];

  cout << "   to solve" << endl;
  // make compressed
  M.makeCompressed();
  Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> > solverA;

  bool flag;
  solverA.compute(M);
  Mat m_du = Mat::zeros(H_, W_, CV_32FC1);
  if(solverA.info() != Eigen::Success)
    {
      cout << "decomposition failed" << endl;
      flag = false;
    }
  else
    {
      
      cout << "solver built" << endl;
      du = solverA.solve(vg);
      if(solverA.info() != Eigen::Success)
	{
	  cout << "solve failed!" << endl;
	  flag = false;
	}
      else
	{
	  cout << "   solved" << endl;
	 
	  flag = true;
	  float *du_data = reinterpret_cast<float*>(m_du.data);
	  for(int i = 0; i < H_; ++i)
	    for(int j = 0; j < W_; ++j)
	      {
		int ind = i * W_ + j;
		du_data[ind] = du(ind);
	      }
	}

    }
  if ( !flag )
    {

      cout << "solver failed" << endl;
      flag = false;
    }
  else 
    {
      Mat inner;
      multiply(m_du, g, inner);
      float up = sum(inner)[0];
      Mat _a, _b;
      multiply(g, g, _a);
      multiply(m_du, m_du, _b);
      float down1 = sqrt(sum(_a)[0]);
      float down2 = sqrt(sum(_b)[0]);
      cout << "-gd/|g||d| = " << -up/down1/down2 << endl;
      if (abs(-up / down1 / down2) < epsilon_d_)
	{
	  cout << "    flag = false" << endl;
	  flag = false;
	}
      else
	{
	  cout << "    flag = true" << endl;
	  flag = true;
	}
    }

  sucess = flag;
  return m_du;
}
void LRTVPHI::sub_1(float beta0_, float sigma0_)
{
  // superlinearly convergent R-regularized Newton Scheme
  float c = 1.0e9;
  float rho1 = 0.25;
  float rho2 = 0.75;
  float k1 = 0.25;
  float k2 = 2;
  float epsilon = 1.0e-4 * lambda_tv_;
  float eps_d = 1.0e-8;
  float tau1 = 0.1;
  float tau2 = 0.9;

  // initialize u0, p0
  float beta0 = beta0_;
  float sigma0 = sigma0_;

  // beta and sigma need initialization
  float beta = beta0;
  float sigma = sigma0;
  float beta_last = 0;
  float beta_last_last = 0;
  float rho;
  int k = 0;


  bool flag;
  float gamma = 0.5;
  // initialization not implemented !!!
  // initialization of U_ is straightforward
  // What about the initialization of px_ and py_
  px_ = Mat::zeros(H_, W_, CV_32FC1);
  py_ = px_.clone();

  // 问题： 这个时候U_x_ 以及 phi_m_ 还没有算出来呢
  Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
  // default border type: BORDER_REFLECT101
  filter2D(U_, grad_x_plus, -1, kernelx_plus_);
  filter2D(U_, grad_x_minus, -1, kernelx_minus_);
  filter2D(U_, grad_y_plus, -1, kernely_plus_);
  filter2D(U_, grad_y_minus, -1, kernely_minus_);

    
  U_x_ = (grad_x_minus + grad_x_plus) / 2.0;
  U_y_ = (grad_y_minus + grad_y_plus) / 2.0;



  Mat U_x2, U_y2;
  // make m
  multiply(U_x_, U_x_, U_x2);
  multiply(U_y_, U_y_, U_y2);

  Mat gradAbs = U_x2 + U_y2;
  sqrt(gradAbs, gradAbs);

  max(gradAbs, gamma_, m_);

  // characteristic matrix 
  threshold(gradAbs, chi_, gamma_, 1.0, THRESH_BINARY);

  m_.copyTo(inverse_phi_m_);

  float *inverse_phi_m_data = reinterpret_cast<float*>(inverse_phi_m_.data);
  for(int i = 0; i < W_ * H_; ++i)
    {
      inverse_phi_m_data[i] = dtrans(inverse_phi_m_data[i]) / inverse_phi_m_data[i];
    }

  multiply(U_x_, inverse_phi_m_, px_);
  multiply(U_y_, inverse_phi_m_, py_);

  Mat d;


  while(1)
    {
      cout << "========================" << endl;
      cout << "outer loop = " << k << endl;
      cout << "NORM TVPHI = " << NORM_TVPHI_h(U_) << endl;
      
      cout << " Objective Function Value = " << sub_1_val(U_) << endl;
      // generate Hk Rk gk at (uk, pk)
      // compute gradients for U_
      Mat g = make_g();

      make_H_R(epsilon);

      // H^k consists of two parts

      Mat part1 = 2.0 * mask_ + rho_;
      int iter = 0;


      while(1)
	{
	  // solve for direction du dp
	  cout << "--------" << endl;
	  cout << " inner loop = " << iter << endl;
	  while(1)
	    {
	      //debug
	      beta = 1;
	      d = solveForDirection(epsilon, beta, g, flag);
	      goto z;
	      if(flag)
		{
		  break;
		}
	      else
		{
		  cout << "set beta = 1" << endl;
		  cout << "return to step 5" << endl;
		  beta = 1;
		}
	    }
	  //	  if(beta == 1) break;
	  cout << "direction found" << endl;

	  Mat temp = laplacian_D(lambda_tv_, epsilon, BX, BY, d);
 
	  temp = temp + epsilon * d;
 	  multiply(d, temp, temp);
	  float dRd = sum(temp)[0];
	  if (abs(beta - 1) < 1e-10 && dRd > sigma * sigma)
	    {
	      // set sigma
	      cout << "set sigma = " << sqrt(dRd) << endl;
	      sigma = sqrt(dRd);
	      cout << "goto step 15" << endl;
	      break;
	    }
	  // update beta
	  beta += 1.0 / c * (dRd - sigma * sigma);
	  cout << "dRd - sigma^2 = " << dRd - pow(sigma,2) << endl;
	  // project beta on [0,1]
	  beta = max(min(beta, 1.f), 0.f);

	  // inner loop stopping criteria
	  // I don't know what criterions should be put here
	  // I need to discuss with Shenming Zhang

	 
	 
	  float err =  abs(beta - beta_last_last);
	  cout << "  err = " << err << endl;
	  if( err < 1.0e-10 && iter > 4)
	    {
	      cout << "  finally beta = " << beta << endl;
	      break;
	    }
	  iter ++;
	  
	  cout << "  beta = " << beta << endl;
	  beta_last_last = beta_last;
	  beta_last = beta;
	}
    z:
      cout << "beta = " << beta << endl;
      float f_u = sub_1_val(U_);
      Mat u_plus_d = U_ + d;
      float f_u_d = sub_1_val(u_plus_d);

      Mat temp ;
      multiply(g, d, temp);

      float gd = sum(temp)[0];
      Mat Hd = laplacian_D(lambda_tv_, epsilon, AX, AY, d);
      Hd = Hd + hessian_Theta;
      multiply(d, Hd, Hd);
      float dHd = sum(Hd)[0];
      rho = (f_u_d - f_u) / (gd + dHd/2.0);

      cout << "gd = " << gd << endl;
      cout << "dHd = " << dHd << endl;
      cout << "f(u+d) - f(u) = " << f_u_d - f_u << endl;
      cout << "rho = " << rho << endl;
      
      // set sigma
      if (rho < rho1)
	sigma = k1 * sigma;
      else if (rho > rho2)
	sigma = k2 * sigma;

      

      float stepsize = 10000.0;
      float tau = 0.5;
      float c_ = 0.5;

      Mat pd;
      Mat grad_f_x = sub_1_grad(U_);
      multiply(d, grad_f_x, pd);
      float m = sum(pd)[0];
      // determine the step size by Wolfe-Powell conditions

      float t = -c_ * m;
      do
	{

	  float f_x = sub_1_val(U_);
	  Mat x1 = U_ + stepsize * d;

	  float f_x_1 = sub_1_val(x1);
	  if( f_x - f_x_1 > 0 /*stepsize * t*/) break;
	  stepsize *= tau;
	  if(stepsize < 1e-55) break;
	}while(1);
      // update uk, pk
      cout << "step size = " << stepsize << endl;
      // u^{k+1}
      #ifdef _DEBUG_
      Mat prev;
      U_.convertTo(prev, CV_8UC1);
      imshow("orig", prev);

      if(stepsize < 1e-55) break;
      #endif
      U_ = U_ + stepsize * d;

      #ifdef _DEBUG_
      Mat show;
      U_.convertTo(show, CV_8UC1);
      imshow("iterate", show);
      waitKey(0);
      #endif
      Mat tempX, tempY;
      multiply(d_phi_m_over_phi_m_, U_x_, tempX);
      multiply(d_phi_m_over_phi_m_, U_y_, tempY);
      multiply(tempX, px_tilde, tempX);
      multiply(tempY, py_tilde, tempY);
      divide(tempX, m_, tempX);
      divide(tempY, m_, tempY);
      tempX = inverse_phi_m_ - tempX;
      tempY = inverse_phi_m_ - tempY;
      
      Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
      // default border type: BORDER_REFLECT101
      filter2D(d, grad_x_plus, -1, kernelx_plus_);
      filter2D(d, grad_x_minus, -1, kernelx_minus_);
      filter2D(d, grad_y_plus, -1, kernely_plus_);
      filter2D(d, grad_y_minus, -1, kernely_minus_);

      Mat d_x, d_y;

      d_x = (grad_x_minus + grad_x_plus) / 2.0;
      d_y = (grad_y_minus + grad_y_plus) / 2.0;


      multiply(tempX, d_x, tempX);
      multiply(tempY, d_y, tempY);

      Mat partX, partY;
      
      // p^{k+1}
      multiply(inverse_phi_m_, U_x_, partX);
      multiply(inverse_phi_m_, U_y_, partY);
      px_ = partX + tempX;
      py_ = partY + tempY;

      cout << "Objective Function Value After = " << sub_1_val(U_) << endl;;
      // outer loop stopping criteria
      // I don't know what stopping criterion to use here
      // Need to discuss with Zhang Shenming
      if( k > 10 ) break;

      cout << "NORM TVPHI after = " << NORM_TVPHI_h(U_) << endl;
      k++;
    }
  
}

Mat LRTVPHI::compute()
{
  int max_iter = 200;
  for(int iter = 0; iter < max_iter; ++iter)
    {
      cout << "=====================" << endl;
      cout << "Iter = " << (iter + 1) << endl;

      sub_1(1.0,10.0);
      sub_2();
      cout << "subproblem 2 done" << endl;
      sub_3();

      // when to stop needs further consideration
      if( iter >= 1 && norm(U_, U_last_) / norm(U_last_) < 1.0e-4 )
	{
	  break;
	}
      cout << "relative error = " << norm(U_, U_last_) / norm(U_last_) << endl;
      cout << endl;
      U_.copyTo(U_last_);


    }

  return U_;
}
