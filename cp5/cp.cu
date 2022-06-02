#include <vector>
#include <numeric>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}






// based on course material
__global__ void mykernel(int ny, int nx, int n, int nn, float* dnor, float* result) {

    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* t = dnor + nn * nn;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = 0;
        }
    }
    for (int k = 0; k < n; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t[nn*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = t[nn*k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] += x[ib] * y[jb];
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) {
                result[j+i*ny] = v[ib][jb];
            }
        }
    }
}



__global__ void mytkernel(int ny, int nx, int nn, float* d, float* vnor) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    float* t = d + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < ny && j < nx) ? vnor[nx * i + j] : 0;
        d[nn * i + j] = v;
        t[nn * j + i] = v;
    }
}

void correlate(int ny, int nx, const float* data, float* result) {

    int n = 0;
    int nn = 0;
    if (ny >= nx) { 
        nn = roundup(ny, 64);
        n = ny;
    } else {
        nn = roundup(nx, 64);
        n = nx;
    }

    std::vector<float> vnor(nx * ny);
    vector<float> rs(ny, 0.0);  //sum per row
    vector<float> rss(ny, 0.0); //sum of square per row
    #pragma omp parallel for
    for(int i=0; i<ny; i++){
        for(int c=0; c<nx; c++){
            rs[i]+=data[c+i*nx];
        }
    }
    
    for(int i=0; i<ny; i++){
        for(int c=0; c<nx; c++){
            vnor[c+i*nx] = data[c+i*nx]-rs[i]/nx;
            rss[i]+=vnor[c+i*nx]*vnor[c+i*nx];
        }
    }
    
    for(int i=0; i<ny; i++){
        for (int c=0; c<nx; c++){
            vnor[c+i*nx]=vnor[c+i*nx]/sqrt(rss[i]);
        }
    }
    

    
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny*ny*sizeof(float)));
    float* dGPU_tmp = NULL;
    CHECK(cudaMalloc((void**)&dGPU_tmp, nx*ny*sizeof(float)));
    CHECK(cudaMemcpy(dGPU_tmp, vnor.data(), nx*ny*sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        mytkernel<<<dimGrid, dimBlock>>>(ny, nx, nn, dGPU, dGPU_tmp);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        mykernel<<<dimGrid, dimBlock>>>(ny, nx, n, nn, dGPU, rGPU);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny*ny*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(dGPU_tmp));
}
