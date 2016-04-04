#ifndef __COMMON_H_
#define __COMMON_H_
#include <map>
#include <list>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define garma 2.2
class element{
 public:
  int w;
  int w_mask;
  float Y;
  float M_mean, Y_mean, I_mean;
  list<int> G,N;
  map<int,int> c;
};

list<int>::iterator ffind(list<int>::iterator start, list<int>::iterator end, int value);

float g(int iter, int k, float l);

map<int, int>::iterator ffind(map<int, int>::iterator start, map<int, int>::iterator end, int value);

#endif
