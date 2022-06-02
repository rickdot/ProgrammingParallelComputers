#include <math.h>
#include <vector>
#include <iostream>




/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 4;
    int na = (nb+nx-1) / nb;
    int nab = na*nb;

    //normalization
    std::vector<double> v_padded(nab*ny, 0); // store normailzed matrix
    double row_sum;
    double mean;
    double x;
    int col;
    double sumf=0;

    
    for(int i=0; i<ny; i++){
        row_sum=0;
        for(int c=0; c<nx; c++){
            x=data[c+i*nx];
            row_sum+=x;
        }
        mean=row_sum/nx;
        for(int c=0; c<nx; c++){
            v_padded[c+i*nab] = data[c+i*nx]-mean;
        }

        row_sum=0;
        for(int c=0; c<nx; c++){
            x=v_padded[c+i*nab];
            row_sum+=x*x;
        }
        for (int c=0; c<nx; c++){
            v_padded[c+i*nab]=v_padded[c+i*nab]/sqrt(row_sum);
        }
    }

    // matrix multiplication
    double sum[nb];
    for(int j=0; j<ny; j++){
        for(int i=j; i<ny; i++){
            for(int k=0; k<nb; k++){
                sum[k] = 0;  // sum[0], sum[1], sum[2], sum[3]
            }
            for(int ka=0; ka<na; ka++){
                for(int kb=0; kb<nb; kb++){
                    col = ka*nb+kb;
                    sum[kb] += v_padded[col+i*nab] * v_padded[col+j*nab];
                }
            }
            sumf = 0;
            for(int k=0; k<nb; k++){
                sumf+=sum[k];
            }
            result[i+j*ny] = float(sumf);

        }
    }

}
