#include <cmath>
#include <vector>
#include <x86intrin.h>

using namespace std;

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

static float8_t* float8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(float8_t), sizeof(float8_t) * n)) {
        throw std::bad_alloc();
    }
    return (float8_t*)tmp;
}

constexpr float8_t f8zero {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

// static inline double sum4d(double4_t vv){
//     double v = 0.0;
//     for(int i=0; i<4; ++i){
//         v += vv[i];
//     }
//     return v;
// }


static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }


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
    vector<float> vnor(ny*nx, 0.0); //normalized data
    vector<float> rs(ny, 0.0);  //sum per row
    vector<float> rss(ny, 0.0); //sum of square per row

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
    int nb = 8; // 8 elements per vector
    int na = (ny+nb-1) / nb;
    float8_t* vd = float8_alloc(nx*na); // nx column, na vectors per column
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
            float8_t vs000 = f8zero;
            float8_t vs001 = f8zero;
            float8_t vs010 = f8zero;
            float8_t vs011 = f8zero;
            float8_t vs100 = f8zero;
            float8_t vs101 = f8zero;
            float8_t vs110 = f8zero;
            float8_t vs111 = f8zero;

            for(int k=0; k<nx; ++k){

                float8_t a000 = vd[nx*ia + k];
                float8_t b000 = vd[nx*ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);

                vs000 = vs000 + (a000 * b000);
                vs001 = vs001 + (a000 * b001);
                vs010 = vs010 + (a010 * b000);
                vs011 = vs011 + (a010 * b001);
                vs100 = vs100 + (a100 * b000);
                vs101 = vs101 + (a100 * b001);
                vs110 = vs110 + (a110 * b000);
                vs111 = vs111 + (a110 * b001);
            }

            float8_t vs[8] = { vs000, vs001, vs010, vs011, vs100, vs101, vs110, vs111 };
            for(int kb=1; kb<nb; kb+=2){
                vs[kb] = swap1(vs[kb]);
            }

            for(int jb=0; jb<nb; ++jb){
                for(int ib=0; ib<nb; ++ib){
                    int i = ib + ia * nb;
                    int j = jb + ja * nb;
                    if(i <= j){
                        if(j < ny && i < ny){
                            result[ny*i + j] = vs[ib^jb][jb];
                        }
                    }
                }
            }
        }
    }
  std::free(vd);

}
