#include <vector>
#include <stdlib.h>
#include <iostream>


using namespace std;

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

constexpr double4_t d4z {
    0.0, 0.0, 0.0, 0.0
};

static inline double sum3c(double4_t sum) {
    double h=0;
    h += sum[0];
    h += sum[1];
    h += sum[2];
    return h;
}

double4_t avg(double4_t sum, double size){
    double4_t r = d4z;
    r[0] = sum[0]/size;
    r[1] = sum[1]/size;
    r[2] = sum[2]/size;
    return r;
}

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    
    // pre-computed array
    double4_t* vda = double4_alloc((ny+1)*(nx+1));
    # pragma omp parallel for schedule(static,1)
    for(int y=0; y<=ny; y++){
        for(int x=0; x<=nx; x++){
            vda[x+y*(nx+1)] = d4z;
        }
    }
    # pragma omp parallel for schedule(static,1)
    for(int y=1; y<=ny; y++){
        for(int x=1; x<=nx; x++){
            for(int c=0; c<3; c++){
                vda[x+y*(nx+1)][c]=data[c+3*(x-1)+3*nx*(y-1)]+vda[(x-1)+y*(nx+1)][c];
            }
        }
    }
    # pragma omp parallel for schedule(static,1)
    for(int x=1; x<=nx; x++){
        for(int y=1; y<=ny; y++){
            vda[x+y*(nx+1)] += vda[x+(y-1)*(nx+1)];
        }
    }

    double4_t psum = vda[nx+ny*(nx+1)]; // sum of all pixels
    double psize = ny*nx;

    

    int r_y0=0; 
    int r_x0=0;
    int r_y1=0;
    int r_x1=0;
    double util_best = -1;
    #pragma omp parallel
    {   
        // not shared
        double util_tmp = -1;
        int x0_tmp = 0;
        int y0_tmp = 0; 
        int x1_tmp = 0; 
        int y1_tmp = 0;
        #pragma omp for schedule(dynamic,1)
        for(int wxy=0; wxy<nx*ny; wxy++){
            
            int wx = wxy%nx+1;
            int wy = wxy/nx+1;
            double wsize = wy * wx;
            double wsize1 = 1/ wsize;
            double bsize1 = 1/(psize-wsize);
            // #pragma omp critical
            // {
            //    cout<<wxy<<' '<<wx<<' '<<wy<<endl;   
            // }
            
            for (int y0 = 0; y0<=ny-wy; y0++){
                for (int x0 = 0; x0<=nx-wx; x0++){
                    int y1 = y0+wy;
                    int x1 = x0+wx;
                    double4_t sumin = vda[x1 + (nx+1) * y1] - vda[x0 + (nx+1) * y1] 
                                    - vda[x1 + (nx+1) * y0] + vda[x0 + (nx+1) * y0];
                    double4_t sumout = psum - sumin;
                    // util = (vin^2)/wsize + (vout^2)/(psize-wsize)
                    double util = sum3c(sumin*sumin*wsize1) 
                                + sum3c(sumout*sumout*bsize1);
                    if (util > util_tmp){
                        util_tmp = util;
                        y0_tmp = y0;
                        x0_tmp = x0;
                        y1_tmp = y1;
                        x1_tmp = x1;
                    }
                }
            }
        }

        // get the best among all threads
        #pragma omp critical
        {
            if (util_tmp > util_best){
                util_best = util_tmp;
                r_y0 = y0_tmp;
                r_x0 = x0_tmp;
                r_y1 = y1_tmp;
                r_x1 = x1_tmp;
            }
        }
    }
    
    double4_t inner_sum = vda[r_x1+(nx+1)*r_y1]-vda[r_x0+(nx+1)*r_y1]
        -vda[r_x1+(nx+1)*r_y0]+vda[r_x0+(nx+1)*r_y0];
    double4_t outer_sum = psum - inner_sum;
    int r_size = (r_y1-r_y0)*(r_x1-r_x0);
    double4_t inner =  avg(inner_sum, r_size);
    double4_t outer =  avg(outer_sum, psize-r_size);

    std::free(vda);

    Result result{r_y0, r_x0, r_y1, r_x1, 
        {(float)outer[0], (float)outer[1], (float)outer[2]},
        {(float)inner[0], (float)inner[1], (float)inner[2]}
    };
    
    
    return result;

}
