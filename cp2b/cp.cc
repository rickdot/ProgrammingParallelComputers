#include <vector>
#include <math.h>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    
    //normalization
    std::vector<double> v(nx*ny,0); // store normailzed matrix
    double sum[ny];
    double mean[ny];

    

    #pragma omp parallel for
    for(int i=0; i<ny; i++){
        sum[i]=0;
        for(int c=0; c<nx; c++){
            sum[i]+=data[c+i*nx];
        }

        mean[i]=sum[i]/nx;
        for(int c=0; c<nx; c++){
            v[c+i*nx] = data[c+i*nx]-mean[i];
        }

        sum[i]=0;
        for(int c=0; c<nx; c++){
            sum[i]+=v[c+i*nx]*v[c+i*nx];
        }

        for (int c=0; c<nx; c++){
            v[c+i*nx]=v[c+i*nx]/sqrt(sum[i]);
        }
    }



    std::vector<double> vsum(ny*ny, 0);
    // matrix multiplication
    // #pragma omp parallel for
    #pragma omp parallel for schedule(static,1)
    for(int j=0; j<ny; j++){
        for(int i=j; i<ny; i++){
            vsum[i+j*ny]=0;
            for(int c=0; c<nx; c++){
                vsum[i+j*ny]+=v[c+i*nx]*v[c+j*nx];
            }
            result[i+j*ny]=float(vsum[i+j*ny]);
        }
    }
}
