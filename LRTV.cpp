#include "LRTV.h"
#include "util.h"

Mat LRTV::getU()
{
  return U_;
}

Mat LRTV::getM()
{
  return M_;
}

Mat LRTV::getY()
{
  return Y_;
}

void LRTV::init_U(Mat &u0)
{
  u0.convertTo(U_, CV_32FC1);
}

// mask: uint8_t type 1 channel taking values 0 and 255
// I: uint8_t grayscale image
LRTV::LRTV(Mat &I, Mat &mask)
{
  W_ = I.cols;
  H_ = I.rows;
  I_ = Mat::zeros(H_, W_, CV_32F);
  mask_ = Mat::zeros(H_, W_, CV_32F);

  I.convertTo(I_, CV_32FC1, 1.0);
  mask.convertTo(mask_, CV_32FC1, 1.0/255.0);
  // initialize
  U_ = Mat::zeros(H_, W_, CV_32F);
  U_last_ = Mat::zeros(H_, W_, CV_32F);
  // internal variables
  M_ = Mat::zeros(H_, W_, CV_32F);
  Y_ = Mat::zeros(H_, W_, CV_32F);

  // make gradient operators
  kernelx_plus_ = (Mat_<float>(1,3)<<0.0,-1.0,1.0);
  kernelx_minus_ = (Mat_<float>(1,3)<<-1.0,1.0,0.0);
  kernely_plus_ = (Mat_<float>(3,1)<<0.0,-1.0,1.0);
  kernely_minus_ = (Mat_<float>(3,1)<<-1.0,1.0,0.0);
}

void LRTV::setParameters(float rho, float dt, float lambda_tv, float lambda_rank)
{
  rho_ = rho;
  dt_ = dt;
  h_ = 1.0;
  alpha_ = 1.0;
  lambda_rank_ = lambda_rank; //10;
  lambda_tv_ = lambda_tv;
  cout << "set parameters done" << endl;
}

// objective function evaluation
float LRTV::sub_1_val(Mat &X)
{
  float val = 0.0;
  Mat part1 = (X - I_);
  multiply(part1, mask_, part1);
  pow(part1, 2.0, part1);

  val += sum(part1)[0];

  val += lambda_tv_ * NORM_TV(X);

  Mat part2 = X - M_ + Y_;
  pow(part2, 2.0, part2);
  part2 = rho_ / 2.0 * part2;

  val += sum(part2)[0];
  return val;
}

void LRTV::sub_1()
{
  float epsilon = 1.0e-4;

  int totalIterations, iter;
  totalIterations = 3000;
  iter = 0;

  float TV, TVlast;
  float objval_last, objval;
  // how to initialize U_
  while(iter < totalIterations)
    {
      //      cout << U_.rows << " " << U_.cols << endl;
      // cout << I_.rows << " " << I_.cols << endl;
      Mat part1 = -2.0  * (U_ - I_);
      multiply(part1, mask_, part1);
      part1 = part1 - rho_ * (U_ - M_ + Y_);

      // debugging purpose
      Mat grad3 = Mat::zeros(H_, W_, CV_32FC1);
      grad3 = rho_ * (U_ - M_ + Y_);

      // compute gradients for U_
      Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
      // default border type: BORDER_REFLECT101
      filter2D(U_, grad_x_plus, -1, kernelx_plus_);
      filter2D(U_, grad_x_minus, -1, kernelx_minus_);
      filter2D(U_, grad_y_plus, -1, kernely_plus_);
      filter2D(U_, grad_y_minus, -1, kernely_minus_);

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
      pow(denominator, 1.5, denominator);

      Mat part2;
      divide(numerator, denominator, part2);
      part2 = lambda_tv_ * part2;
      
      // line search for step_size
      float step_size = dt_;
      float tau = 0.8;
      float c = 0.25;

      // cout << "gradient 1 = " << norm(part1, NORM_L2) << endl;
      // cout << "gradient 2 = " << norm(part2, NORM_L2) << endl;
      // cout << "gradient 3 = " << norm(grad3, NORM_L2) << endl;
      //
      //part1 = -part1;
      //

      float f_0 = sub_1_val(U_);
      while(1)
	{
	  Mat U_new = U_ + step_size * (part1 + part2);
	  
	  float f_1 = sub_1_val(U_new);
	  Mat m;
	  multiply(part1+part2,part1+part2,m);
	  float m_ = sum(m)[0] / W_ / H_;
	  float t = m_ * c;

	  if(f_0-f_1 > step_size * t)
	    {
	      break;
	    }
	  step_size *= tau;
	  if(step_size < 1e-15)
	    {
	      cout << "Ooooooops! Fail to find descending direction" << endl;
	      return;
	    }
	}
      //      cout << "step size= " << step_size << endl;
      Mat U_1 = U_ + step_size * (part1 + part2);
      
      iter ++;
      // cout << "iteration: " << iter << endl;
      //      cout << "   error = " << norm(U_1-U_) << endl;

      //
      TVlast = NORM_TV(U_);
      TV = NORM_TV(U_1);
      objval = sub_1_val(U_1);
      objval_last = sub_1_val(U_);
      // when to stop iteration needs further consideration

      cout << objval << endl;
      cout << "error = " << abs(objval - objval_last) / objval_last << endl;
      cout << ((objval - objval_last)>0?"ascending":"descending") << endl;
      if( iter >=20 && abs(objval - objval_last) / objval_last < 1e-3)
      	{
      	  cout <<"TV = (" << TVlast << ") -> " << TV << endl;
      	  cout << "objval = (" << objval_last << ") -> " << objval << endl;
      	  break;
      	}
      
      // cout << "TV = " << TVlast << " -> " << TV  << endl;
      // cout << "objval = " << objval_last << " -> " << objval << endl;
      U_ = U_1;
      objval_last = objval;
      
    }
}


void LRTV::sub_2()
{
  Mat A = U_ + Y_;
  Mat mask = 255*Mat::ones(H_, W_, CV_8UC1);
  float lambda = rho_ / 2.0 / lambda_rank_ / alpha_;
  
  // TNNR
  Mat At;
  A.convertTo(At, CV_8UC1);
  M_ = TNNR(At, mask, 9, 9, lambda);
  
  
}

void LRTV::sub_3()
{
  Y_ = Y_ + (U_ - M_);
}

Mat LRTV::compute()
{
  int max_iter = 30;
  for(int iter = 0; iter < max_iter; ++iter)
    {
      cout << "=====================" << endl;
      cout << "Iter = " << (iter + 1) << endl;
      
      sub_1();
      sub_2();
      sub_3();

      // when to stop needs further consideration
      if( iter >= 5 && norm(U_, U_last_) / norm(U_last_) < 1.5e-3 )
	{
	  break;
	}
      cout << "relative error = " << 1.0* norm(U_, U_last_) / norm(U_last_) << endl;
      U_.copyTo(U_last_);

      
      cout << "PSNR = " << PSNR(I_, U_, mask_) << endl;
      cout << endl;
      
    }

  return U_;
}
