#include "tnnr.h"

inline float traceNorm(Mat &X)
{
  Mat w;
  SVD::compute(X,w);
  return sum(w)[0];
}
// mask 8-bit 1 channel 0 and 255
// other matrices must be of CV_32FC1 type
Mat APGL(Mat &A, Mat &B, Mat &X, Mat &M, Mat &mask, float eps, float lambda)
{
  Mat AB = A.t() * B;
  Mat BA = B.t() * A;

  int W = X.cols;
  int H = X.rows;
  
  Mat lastX = Mat::zeros(H, W, CV_32FC1);
  Mat Y = Mat::zeros(H, W, CV_32FC1);
  X.copyTo(lastX);
  lastX.copyTo(Y);


  float tlast = 1.0;
  float t = 1.0;
  Mat XX = Mat::zeros(H, W, CV_32FC1);
  X.copyTo(XX);

  Mat known;
  mask.convertTo(known, CV_32FC1, 1.0/255.0);


  Mat u, sigma, v;

  float objval = 0.0;
  float objval_last = 0.0;
  int k = 1;

  for(k = 1; k < 201; k++)
    {
      Mat tmp;
      multiply(Y-M, known, tmp);
      Mat G = Y + t * (AB - lambda * tmp);
      //svd for G
      SVD::compute(G, sigma, u, v);

      sigma = max(sigma - t, 0.0);
      sigma = Mat::diag(sigma);
      X = u*sigma*v;
      t = (1 + sqrt(1+4*tlast*tlast))/2;
      Y = X + (tlast-1)/(t)*(X - lastX);

      X.copyTo(XX);
      X.copyTo(lastX);
      tlast = t;

      multiply(X-M, known, tmp);
      float tr = trace(X*BA)[0];
      objval = traceNorm(X)  - tr + lambda / 2.0 * pow(norm(tmp,NORM_L2),2); 
      if(k >= 2 && -(objval - objval_last) < eps)
	{
	  break;
	}

      objval_last = objval;
    }

  //  cout << "APGL total Iterations = " << k << endl;
  return XX;
}


Mat TNNR(Mat &im0, Mat &mask, int lower_R, int upper_R, float lambda)
{
  Mat X, M, A, B;

  im0.convertTo(X, CV_32FC1);
  X.copyTo(M);

  int W = X.cols;
  int H = X.rows;
  Mat X_rec = Mat::zeros(H, W, CV_32FC1);
  Mat X_rec_last = Mat::zeros(H, W, CV_32FC1);
  float eps = 0.1;
  int number_out_of_iter = 10;

  
  for(int R = lower_R; R <= upper_R; R++)
    {

      for(int out_iter = 1; out_iter < 11; ++out_iter)
	{
	  //	  cout << "  TNNR  iter = " << out_iter << endl;
	  Mat u,sigma,v;
	  SVD::compute(X, sigma, u, v);
	  A = u(Range::all(), Range(0, R - 1));
	  A = A.t();
	  B = v(Range(0, R - 1), Range::all());

	  X_rec = APGL(A, B, X, M, mask, eps, lambda);

	  if(out_iter >= 2 && norm(X_rec - X_rec_last) / norm(M,NORM_L2) < .01)
	    {
	      X_rec.copyTo(X);
	      break;
	    }

	  X_rec.copyTo(X);
	  X_rec.copyTo(X_rec_last);

	     
	}
    } 

  return X;
}
