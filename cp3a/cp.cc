#include <cmath>
#include <vector>
#include <x86intrin.h>

using namespace std;

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

constexpr double4_t d4zero {
    0.0, 0.0, 0.0, 0.0
};

// static inline double sum4d(double4_t vv){
//     double v = 0.0;
//     for(int i=0; i<4; ++i){
//         v += vv[i];
//     }
//     return v;
// }


static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 0b00000001); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b00000101); }


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    // normalization
    // by row
    vector<double> vnor(ny*nx, 0.0); //normalized data
    vector<double> rs(ny, 0.0);  //sum per row
    vector<double> rss(ny, 0.0); //sum of square per row

    #pragma omp parallel for
    for(int i=0; i<ny; i++){
        for(int c=0; c<nx; c++){
            rs[i]+=data[c+i*nx];
        }
    }
    #pragma omp parallel for
    for(int i=0; i<ny; i++){
        for(int c=0; c<nx; c++){
            vnor[c+i*nx] = data[c+i*nx]-rs[i]/nx;
            rss[i]+=vnor[c+i*nx]*vnor[c+i*nx];
        }
    }
    #pragma omp parallel for
    for(int i=0; i<ny; i++){
        for (int c=0; c<nx; c++){
            vnor[c+i*nx]=vnor[c+i*nx]/sqrt(rss[i]);
        }
    }

    //vectorize by column
    int nb = 4; // 4 elements per vector
    int na = (ny+nb-1) / nb;
    double4_t* vd = double4_alloc(nx*na); // nx column, na vectors per column
    #pragma omp parallel for
    for(int i=0; i<nx; ++i){
        for(int ka=0; ka<na; ++ka){
            for(int kb=0; kb<nb; ++kb){
                int j=ka*nb+kb;
                vd[nx*ka+i][kb] = (j < ny) ? (vnor[nx*j+i]) : (0.0);
            }
        }
    }

   
    #pragma omp parallel for schedule(static, 1)
    for(int ia=0; ia<na; ++ia){
        for(int ja=ia; ja<na; ++ja){
            double4_t vs00 = d4zero;
            double4_t vs01 = d4zero;
            double4_t vs10 = d4zero;
            double4_t vs11 = d4zero;

            for(int k=0; k<nx; ++k){
                double4_t a00 = vd[nx*ia + k];
                double4_t b00 = vd[nx*ja + k];
                double4_t a10 = swap2(a00);
                double4_t b01 = swap1(b00);

                vs00 = vs00 + (a00 * b00);
                vs01 = vs01 + (a00 * b01);
                vs10 = vs10 + (a10 * b00);
                vs11 = vs11 + (a10 * b01);
            }

            double4_t vs[4] = {vs00, vs01, vs10, vs11}; 
            for(int kb=1; kb<nb; kb+=2){
                vs[kb] = swap1(vs[kb]);
            }

            for(int jb=0; jb<nb; ++jb){
                for(int ib=0; ib<nb; ++ib){
                    int i=ib+ia*nb;
                    int j=jb+ja*nb;
                    if(i<=j){
                        if(j<ny && i<ny){
                            result[ny*i+j] = vs[ib^jb][jb];
                        }
                    }
                }
            }
        }
    }
    free(vd);



}
