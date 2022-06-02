#include <vector>
#include <cmath>


// define vector to store 4 doubles
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
// to guarantee aligned memory allocation
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}
// helper functions
constexpr double4_t d4z{
    0.0, 0.0, 0.0, 0.0
};

static inline double sum_4(double4_t vv){
    double v = 0.0;
    for(int i=0; i<4; ++i){
        v += vv[i];
    }
    return v;
}


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    int na = (nb+nx-1) / nb;

    

    // input data, padded, converted to vectors
    double4_t* vd = double4_alloc(ny*na);
    // std::vector<double> v_padded(nab*ny, 0);

    //normalization
    std::vector<double> v(nx*ny, 0); // store normailzed matrix
    double sum;
    double mean;
    
    for(int i=0; i<ny; i++){
        sum=0;
        for(int c=0; c<nx; c++){
            sum+=data[c+i*nx];
        }
        mean=sum/nx;
        for(int c=0; c<nx; c++){
            v[c+i*nx] = data[c+i*nx]-mean;
        }

        sum=0;
        for(int c=0; c<nx; c++){
            sum+=v[c+i*nx]*v[c+i*nx];
        }
        for (int c=0; c<nx; c++){
            v[c+i*nx]=v[c+i*nx]/sqrt(sum);
        }


        for(int j=0; j<na; j++){
            for(int k=0; k<nb; k++){
                vd[j+na*i][k] = (j*nb+k < nx) ? (v[nx*i+j*nb+k]) : 0.0;
            }
        }

    }


    double4_t v_prod;

    // matrix multiplication
    for(int j=0; j<ny; j++){
        for(int i=j; i<ny; i++){
            v_prod = d4z;
            for(int vk=0; vk<na; vk++){
                v_prod += vd[vk+na*j] * vd[vk+na*i];
            }
            result[i+j*ny] = sum_4(v_prod);
        }
    }


    std::free(vd);
}
