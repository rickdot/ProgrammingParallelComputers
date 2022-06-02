#include <iostream>
#include <vector>
#include <algorithm>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

  std::vector<float> v = {};
  double result = 0;
  // iterate over all pixels
  for(int x=0; x<nx; x++){
    for(int y=0; y<ny; y++){
      result = 0;
      v = {};
      // find all pixels within each window      
      for(int i=x-hx; i<=x+hx; i++){
        for(int j=y-hy; j<=y+hy; j++){
          if (i>=0 && i<nx){
            if (j>=0 && j<ny){
              v.push_back(in[i+j*nx]);
            }
          }
        }
      }

      //find the median
      int n = v.size();
      if(n % 2 == 0){ //even
        nth_element(v.begin(), v.begin() + n / 2, v.end());
        double a = v[n/2];
        nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        double b = v[n/2-1];
        result = (double)(a+b) / 2.0;
      } else { //odd
        nth_element(v.begin(), v.begin() + n / 2, v.end());
        result = (double)v[n / 2];
      }
      
      out[x + nx*y] = result;
    }
  }


}
