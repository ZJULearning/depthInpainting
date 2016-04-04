#include <opencv2/opencv.hpp>
#include <iostream>
#include "util.h"
#include "LRTV.h"
#include "tnnr.h"
#include <fstream>
#include "LRTVPHI.h"
#include "LRL0.hpp"
#include "LRL0PHI.hpp"
#include <numeric>
using namespace cv;
using namespace std;


int main(int argc, char ** argv)
{
  if(argc < 2)
    {
      cout << "Usage: " << endl;
      cout << "         TV norm: ./depthInpainting TV depthImage" << endl;
      cout << "       PSNR calc: ./depthInpainting P depthImage mask inpainted" << endl;
      cout << "      Inpainting: ./depthInpainting LRTV depthImage mask outputPath" << endl;
      cout << "      Generating: ./depthInpainting G depthImage missingRate outputMask outputMissing" << endl;
      cout << "         LowRank: ./depthInpainting L depthImahe mask outputpath" << endl;
      cout << "         LRTVPHI: ./depthInpainting LRTVPHI depthImage mask outputPath" << endl;
      cout << "      TVPHI norm: ./depthInpainting TVPHI depthImage" << endl;
      cout << "            LRL0: ./depthInpainting LRL0 depthImage mask outputPath initImage K lambda_L0 MaxIterCnt" << endl;
      cout << "         LRL0PHI: ./depthInpainting LRL0PHI depthImage mask outputPath initImage K lambda_L0 MaxIterCnt" << endl;
      cout << "              L0: /depthInpainting L0 depthImage" << endl;
      return 0;
    }
  string instruction = argv[1];
  if( instruction == "TV" )
    {
      Mat img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
      cout << NORM_TV(img) << endl;
    }
  if( instruction == "TVPHI")
    {
      Mat img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
      LRTVPHI lrttv;
      lrttv.setShrinkageParam(1.0, 1.0);
      cout << lrttv.NORM_TVPHI(img) << endl;
    }
  else if( instruction == "P" )
    {
      Mat original = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
      Mat inpainted = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);
      Mat mask = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
      cout << PSNR(original, inpainted, mask) << endl;
    }
  else if( instruction == "G" )
    {
      string disparityPath = argv[2];
      int missingRate = atof(argv[3]);
      string outputMask = argv[4];
      string outputMissing = argv[5];
      Mat disparityOriginal = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
 
      int W = disparityOriginal.cols;
      int H = disparityOriginal.rows;

      RNG rng;
      // mask: 0 indicates missing pixels
      Mat mask = Mat::zeros(H, W, CV_8U);
      rng.fill(mask, RNG::UNIFORM, 0, 255);
      threshold(mask, mask, 255 * missingRate / 100, 255, THRESH_BINARY);
      Mat disparityMissing;
      multiply(disparityOriginal, mask / 255, disparityMissing);

      imwrite(outputMask, mask);
      imwrite(outputMissing, disparityMissing);
    }
  // Low rank
  else if(instruction == "L" )
    {
      string disparityPath = argv[2];
      string maskPath = argv[3];
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      string inpaintedPath = argv[4];
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      multiply(disparityMissing, mask / 255, disparityMissing);
      Mat inpainted = TNNR(disparityMissing, mask, 9, 9, 0.06);
      imwrite(argv[4], inpainted);
      
    }
  // only TV inpainting
  else if( instruction == "T" )
    {
      string disparityPath = argv[2];
      string maskPath = argv[3];
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      string inpaintedPath = argv[4];
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      multiply(disparityMissing, mask / 255, disparityMissing);
      // Mat denoised;
      // inpaint(disparityMissing, 255 - mask, denoised, 11, INPAINT_TELEA);

      // imwrite(inpaintedPath, denoised);

      // try our TV
      // we can use low rank to initialize U_
      Mat denoised = imread(argv[5], CV_LOAD_IMAGE_ANYCOLOR); //TNNR(disparityMissing, mask, 9, 9, 0.06);

      LRTV lrtv(disparityMissing, mask);
      // rho dt lambda_tv lambda_rank 100 10
      lrtv.setParameters(0, 1, 0.01, 0);
      lrtv.init_U(denoised);
      lrtv.sub_1();
      Mat M = lrtv.getM();
      Mat Y = lrtv.getY();
      Mat result = lrtv.getU();

      Mat output;
      result.convertTo(output, CV_8UC1);
      imwrite(inpaintedPath, output);
      imwrite("M.png", M);
      imwrite("Y.png", Y);
    }
  else if( instruction == "LRTV" )
    {
      string disparityPath = argv[2];
      string maskPath = argv[3];
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      string inpaintedPath = argv[4];
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      multiply(disparityMissing, mask / 255, disparityMissing);

      // Mat denoised;
      // inpaint(disparityMissing, 255 - mask, denoised, 11, INPAINT_TELEA);

      // imwrite(inpaintedPath, denoised);

      // try our TV
      // we can use low rank to initialize U_
      //Mat denoised = TNNR(disparityMissing, mask, 9, 9, 0.06);
      Mat denoised = imread(argv[5], CV_LOAD_IMAGE_GRAYSCALE);

      LRTV lrtv(disparityMissing, mask);
      // rho dt lambda_tv lambda_rank 100 10
      lrtv.setParameters(1.2, 0.1, 40, 10);
      lrtv.init_U(denoised);
      Mat result = lrtv.compute();
      Mat M = lrtv.getM();
      Mat Y = lrtv.getY();

      
      Mat output;
      result.convertTo(output, CV_8UC1);
      imwrite(inpaintedPath, output);
      imwrite("M.png", M);
      imwrite("Y.png", Y);

    }
  // for stat
  else if( instruction == "S" )
    {
      Mat dispU = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
      int H = dispU.rows;
      int W = dispU.cols;
      Mat disp = Mat::zeros(H, W, CV_32FC1);
      dispU.convertTo(disp, CV_32FC1);
      
      Mat kernelx_plus_ = (Mat_<float>(1,3)<<0.0,-1.0,1.0);
      Mat kernelx_minus_ = (Mat_<float>(1,3)<<-1.0,1.0,0.0);
      Mat kernely_plus_ = (Mat_<float>(3,1)<<0.0,-1.0,1.0);
      Mat kernely_minus_ = (Mat_<float>(3,1)<<-1.0,1.0,0.0);

      Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
      filter2D(disp, grad_x_plus, -1, kernelx_plus_);
      filter2D(disp, grad_x_minus, -1, kernelx_minus_);
      filter2D(disp, grad_y_plus, -1, kernely_plus_);
      filter2D(disp, grad_y_minus, -1, kernely_minus_);
      Mat ux, uy;
      ux = (grad_x_minus + grad_x_plus) / 2.0;
      uy = (grad_y_minus + grad_y_plus) / 2.0;
      pow(ux, 2.0, ux);
      pow(uy, 2.0, uy);
      Mat grad;
      sqrt(ux + uy, grad);
      Mat r;
      grad.convertTo(r, CV_8UC1);
      vector <float> histogram(200,0);
      for(int i = 0; i < H; i++)
	for(int j = 0; j < W; j++)
	  {
	    if(r.at<uchar>(i,j) < 200)
	      {
		int id = r.at<uchar>(i,j);
		histogram[id] = histogram[id] + 1;
	      }
	  }
      
      float total = accumulate(histogram.begin(), histogram.end(), 0.0);
      fstream file(argv[3],ios::out);
      int i = 0;
      for (auto &x : histogram)
	{
	  i++;
	  cout << x << endl;
	  file << i  << " " << x / total << endl;
	}
    }
    else if( instruction == "L0" )
    {
      Mat dispU = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
      int H = dispU.rows;
      int W = dispU.cols;
      Mat disp = Mat::zeros(H, W, CV_32FC1);
      dispU.convertTo(disp, CV_32FC1);
      
      Mat kernelx_plus_ = (Mat_<float>(1,3)<<0.0,-1.0,1.0);
      Mat kernelx_minus_ = (Mat_<float>(1,3)<<-1.0,1.0,0.0);
      Mat kernely_plus_ = (Mat_<float>(3,1)<<0.0,-1.0,1.0);
      Mat kernely_minus_ = (Mat_<float>(3,1)<<-1.0,1.0,0.0);

      Mat grad_x_plus, grad_x_minus, grad_y_plus, grad_y_minus;
      filter2D(disp, grad_x_plus, -1, kernelx_plus_);
      filter2D(disp, grad_x_minus, -1, kernelx_minus_);
      filter2D(disp, grad_y_plus, -1, kernely_plus_);
      filter2D(disp, grad_y_minus, -1, kernely_minus_);
      Mat ux, uy;
      ux = (grad_x_minus + grad_x_plus) / 2.0;
      uy = (grad_y_minus + grad_y_plus) / 2.0;

      Mat grad;
      grad = abs(ux) + abs(uy);
      Mat r;
      grad.convertTo(r, CV_8UC1);
      int l0 = 0;
      for(int i = 0; i < H; i++)
	for(int j = 0; j < W; j++)
	  {
	    if (grad.at<uchar>(i,j ) <1)
	      l0++;
	  }
      cout << W*H - l0 << endl;
    }
  else if( instruction == "LRTVPHI" )
    {
      string disparityPath = argv[2]; //"../../MiddInpaint/ArtL/disp.png";
      string maskPath = argv[3]; //"../../MiddInpaint/ArtL/mask_50.png";
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      string inpaintedPath = argv[4]; //"just_a_test.png";
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      // int H_ = mask.rows;
      // int W_ = mask.cols;
      // mask = 255*Mat::ones(H_, W_, CV_8UC1);
      Mat orig;
      disparityMissing.copyTo(orig); orig.convertTo(orig, CV_32FC1);
      multiply(disparityMissing, mask / 255, disparityMissing);

      imshow("dd",disparityMissing);waitKey(0);
      LRTVPHI lrttv(disparityMissing, mask);
      lrttv.setShrinkageParam(1.0,1.0);

      Mat denoised = imread(argv[5], CV_LOAD_IMAGE_GRAYSCALE);
     
      // rho dt lambda_tv lambda_rank
      // rho = 1.2
      lrttv.setParameters(1.2, 0.1, 10, 0.01);
      lrttv.setNewtonParameters(0.001); // set gamma

      // imshow("temp", denoised);
      // waitKey(0);2
      lrttv.init_U(denoised);
      lrttv.init_M(denoised);
      cout << "optimal objective function value = " << lrttv.sub_1_val(orig) << endl;
      Mat result;
      
	    //      lrttv.sub_2();
	    //      lrttv.sub_3();

      lrttv.sub_1(1.0,10.0);
      cout << "optimal objective function value = " << lrttv.sub_1_val(orig) << endl;
      // cout << "OK  sub_1" << endl;
      //Mat result = lrttv.compute();
      Mat M = lrttv.getM();
      Mat Y = lrttv.getY();
      Mat output;
      result = lrttv.getU();
      result.convertTo(output, CV_8UC1);
      imwrite(inpaintedPath, output);
      imwrite("tM.png", M);
      imwrite("tY.png", Y);
    }
  else if( instruction == "X" )
    {
      Mat a = (Mat_<float>(2,2)<<0,1,2,3);
      Mat x;
      exp(a,x);
      Mat y;
      exp(-a,y);
      Mat result;
      divide(x-y,x+y,result);
      cout << result << endl;
      
    }
  else if( instruction == "LRL0" )
    {
      string disparityPath = argv[2]; //"../../MiddInpaint/ArtL/disp.png";
      string maskPath = argv[3]; //"../../MiddInpaint/ArtL/mask_50.png";
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      Mat orig;
      disparityMissing.copyTo(orig);
      string inpaintedPath = argv[4]; //"just_a_test.png";
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      int K = atoi(argv[6]);
      float lambda_l0 = atof(argv[7]);
      int max_iter = atoi(argv[8]);

      string path1 = argv[9];
      multiply(disparityMissing, mask / 255, disparityMissing);
      Mat denoised = imread(argv[5], CV_LOAD_IMAGE_GRAYSCALE);
      LRL0 lrl0(disparityMissing, mask);
      lrl0.setParameters(1.2,0.1,lambda_l0,10);
      lrl0.init_U(denoised);
      Mat result = lrl0.compute(K, max_iter, inpaintedPath, orig, path1);
      // Mat output;
      //result.convertTo(output, CV_8UC1);
      //imwrite(inpaintedPath, output);
      
      
    }
    else if( instruction == "LRL0PHI" )
    {
      string disparityPath = argv[2]; //"../../MiddInpaint/ArtL/disp.png";
      string maskPath = argv[3]; //"../../MiddInpaint/ArtL/mask_50.png";
      Mat disparityMissing = imread(disparityPath, CV_LOAD_IMAGE_GRAYSCALE);
      Mat orig;
      disparityMissing.copyTo(orig);
      string inpaintedPath = argv[4]; //"just_a_test.png";
      Mat mask = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);

      int K = atoi(argv[6]);
      float lambda_l0 = atof(argv[7]);
      int max_iter = atoi(argv[8]);

      string path1 = argv[9];
      multiply(disparityMissing, mask / 255, disparityMissing);
      Mat denoised = imread(argv[5], CV_LOAD_IMAGE_GRAYSCALE);
      LRL0PHI lrl0phi(disparityMissing, mask);
      lrl0phi.setParameters(1.2,0.1,lambda_l0,10,0.75);
      lrl0phi.init_U(denoised);
      Mat result = lrl0phi.compute(K, max_iter, inpaintedPath, orig, path1);
      
    }
  else
    {
      cout << "Argument Error!" << endl;
      cout << argv[1] << endl;
    }
  return 0;
}
