#include "common.h"
list<int>::iterator ffind(list<int>::iterator start, list<int>::iterator end, int value)
{
  list<int>::iterator p = start;
  while(p != end)
    {
      if(*p == value)
	return p;
      ++p;
    }
  return p;
}

float g(int iter, int k, float l)
{
  float ans = pow(((float)iter/k), garma) * l;
  return ans;
}
map<int, int>::iterator ffind(map<int, int>::iterator start, map<int, int>::iterator end, int value)
{
  map<int, int>::iterator p = start;
  while(p != end)
    {
      if(p->first == value)
	return p;
      ++p;
    }
  return p;
}


