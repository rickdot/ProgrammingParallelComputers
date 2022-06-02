#include <math.h>
#include <iostream>

/*
Implement a simple sequential baseline solution. 
Do not try to use any form of parallelism yet; 
try to make it work correctly first. 
Please do all arithmetic with double-precision floating point numbers.
*/

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    double sum_i, sum_j, sum_ij = 0;
    double square_i, square_j = 0;

    for(int row_j = 0; row_j < ny; row_j++){
        for(int row_i = row_j; row_i < ny; row_i++){
            // column
            sum_i = 0;
            sum_j = 0;
            sum_ij = 0;
            square_i = 0;
            square_j = 0;
            for(int col = 0; col < nx; col++){
                double x = data[col + row_i*nx];
                double y = data[col + row_j*nx];
                sum_i += x;
                sum_j += y;
                sum_ij += x*y;
                square_i += x*x;
                square_j += y*y;
            }
            double corr = (nx * sum_ij - sum_i * sum_j) / sqrt((nx * square_i - sum_i * sum_i)  * (nx * square_j - sum_j * sum_j));

            result[row_i+row_j*ny] = float(corr);
        }
    }



}
