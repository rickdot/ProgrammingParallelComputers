#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

void prt(vector<double> vv){
  for(auto i:vv){
    cout<<i<<' ';
  }
  cout<<'\n';
}


void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

  std::vector<double> v[nx]; // store all pixels within a window
  double result[nx];


  // iterate over all pixels
  #pragma omp parallel for
  for(int x=0; x<nx; x++){
    for(int y=0; y<ny; y++){
      result[x] = 0;
      v[x] = {};

      // find all pixels within each window      
      for(int i=x-hx; i<=x+hx; i++){
        for(int j=y-hy; j<=y+hy; j++){
          if (i>=0 && i<nx){
            if (j>=0 && j<ny){
              v[x].push_back(in[i+j*nx]);
              
            }
          }
        }
      }

      // cout<<x<<' '<<y<<' '<<'\n';
      // cout<<"px=";
      // cout<<in[x+y*nx]<<'\n';
      // prt(v[x]);
      

      //find the median
      std::vector<double> tmpv;
      tmpv = v[x];
      int n = tmpv.size();
      if(n % 2 == 0){ //even
        nth_element(tmpv.begin(), tmpv.begin() + n / 2, tmpv.end());
        double a = tmpv[n/2];
        nth_element(tmpv.begin(), tmpv.begin() + n / 2 - 1, tmpv.end());
        double b = tmpv[n/2-1];
        result[x] = (double)(a+b) / 2.0;
      } else { //odd
        nth_element(tmpv.begin(), tmpv.begin() + n / 2, tmpv.end());
        result[x] = (double)tmpv[n / 2];
      }
      
      out[x + nx*y] = result[x];
    }
  }


}
