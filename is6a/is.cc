#include <vector>
#include "vector.h"
#include <x86intrin.h>

using namespace std;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

// pack 8 consecutive elements into float8_t
// float8_t get8(const float* data, int nx, int y, int x){
//     int start = x+y*(nx+1);
//     float8_t r = {
//         data[start], data[start+1], data[start+2], data[start+3],
//         data[start+4], data[start+5], data[start+6], data[start+7]
//     };
//     return r;
// }

float8_t max8(float8_t x, float8_t y){ 
    return x > y ? x : y;
}

float hmax(float8_t v){
    float r = 0;
    for(int i=0; i<8; ++i){
        if(v[i]>r){
            r = v[i];
        }
    } 
    return r;
}


Result segment(int ny, int nx, const float* data){
    // pre compute sums of rectangles
    vector<float> vda((nx+1)*(ny+1), 0.0);
    # pragma omp parallel for schedule(dynamic,1)
    for(int y=1; y<=ny; y++){
        for(int x=1; x<=nx; x++){
                vda[x+y*(nx+1)]=data[3*(x-1)+3*nx*(y-1)]+vda[(x-1)+y*(nx+1)];
        }
    }
    # pragma omp parallel for schedule(dynamic,1)
    for(int x=1; x<=nx; x++){
        for(int y=1; y<=ny; y++){
            vda[x+y*(nx+1)] += vda[x+(y-1)*(nx+1)];
        }
    }


    int psize = ny*nx;
    float sum_all = vda[nx+ny*(nx+1)];
    float8_t f8psum = {
        sum_all, sum_all, sum_all, sum_all, 
        sum_all, sum_all, sum_all, sum_all
    };

    const float* vda2 = vda.data();

    // best size
    int r_wy = 1;
    int r_wx = 1;

    // find rect size with max utility
    float util_best = -1;
    #pragma omp parallel for schedule(dynamic, 1)
    for(int wxy=0; wxy<nx*ny; wxy++){
        int wx = wxy%nx+1;
        int wy = wxy/nx+1;
        float wsize = wy * wx;
        float wsize1 = 1 / wsize;
        float bsize1 = 1 / (psize - wsize);
        
        float8_t f8max = float8_0;
        float best = -1;
        // each row = some blocks(float8_t) + some floats (<8)
        for(int y0=0; y0<=ny-wy; y0++){
            int y1 = y0 + wy;

            int x0bound = nx - wx + 1;
            int blocks = x0bound / 8;
            // vector operations in blocks
            for(int block=0; block<blocks; block++){
                int x0_start = 8*block;
                int x1_start = x0_start + wx;
                
                float8_t s1 = _mm256_loadu_ps(vda2 + y1*(nx+1) + x1_start);
                float8_t s2 = _mm256_loadu_ps(vda2 + y1*(nx+1) + x0_start);
                float8_t s3 = _mm256_loadu_ps(vda2 + y0*(nx+1) + x1_start);
                float8_t s4 = _mm256_loadu_ps(vda2 + y0*(nx+1) + x0_start);

                float8_t vin = s1 - s2 - s3 + s4;
                float8_t vout = f8psum - vin;
                float8_t v = vin*vin*wsize1 + vout*vout*bsize1;
                f8max = max8(f8max, v);
            }

            // scalar operations
            for(int x0=8*blocks; x0<x0bound; ++x0){
                int x1 = x0+wx;

                float sumin = vda[y1*(nx+1) + x1] - vda[y1*(nx+1) + x0]
                        - vda[y0*(nx+1) + x1] + vda[y0*(nx+1) + x0];
                float sumout = sum_all - sumin;
                float util = sumin*sumin*wsize1 + sumout*sumout*bsize1;
                if(util > best){
                    best = util;
                }   
            }
        }
        // compare vecter max and scalar max
        float vmax = hmax(f8max);
        if(vmax>best){
            best = vmax;
        }

        #pragma omp critical
        {
            if(best > util_best){ 
                util_best = best; 
                r_wx = wx; 
                r_wy = wy; 
            }
        }
    }
    

    int r_y0 = 0;
    int r_x0 = 0;
    int r_y1 = 0;
    int r_x1 = 0;
    float inner = 0;
    float outer = 0;


    float wsize = r_wx*r_wy;
    float wsize1 = 1 / wsize;
    float bsize1 = 1 / (psize - wsize);
    // find best location of the rectangle
    util_best = -1;
    # pragma omp parallel for schedule(dynamic,1)
    for(int y0=0; y0<=ny-r_wy; y0++){
        for(int x0=0; x0<=nx-r_wx; x0++){
            int y1 = y0+r_wy;
            int x1 = x0+r_wx;

            float sumin = vda[y1*(nx+1) + x1] - vda[y1*(nx+1) + x0] 
                - vda[y0*(nx+1) + x1] + vda[y0*(nx+1) + x0];
            float sumout = sum_all - sumin;
            float util = sumin*sumin*wsize1 + sumout*sumout*bsize1;

            #pragma omp critical
            {
                if(util > util_best){
                    util_best = util;
                    r_y0 = y0;
                    r_x0 = x0;
                    r_y1 = y1;
                    r_x1 = x1;
                    inner = sumin * wsize1;
                    outer = sumout * bsize1;
                }
            }
        }
    }

    Result result = { 
        r_y0, r_x0, r_y1, r_x1, 
        {outer, outer, outer}, 
        {inner, inner, inner} 
    };
    return result;
}
