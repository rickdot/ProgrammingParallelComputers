#include <cmath>
#include <vector>

// copied from course material
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}


using namespace std;


__global__ void mykernel(int ny, int nx, float *data, float *result) {
    int r1 = threadIdx.x + blockIdx.x * blockDim.x;
    int r2 = threadIdx.y + blockIdx.y * blockDim.y;
    if (r1 >= ny|| r2 >= ny)
        return;
    float sum = 0;
    for (int col = 0; col < nx; col++)
    {   
        sum +=  data[col+r1*nx] * data[col+r2*nx];
    }
    result[r1+r2*ny] = sum;
    
}


// ny * ny * nx units of work
// each thread   1*1*nx
// each block 16*16 threads   16*16*nx units of work
// number of blocks = (ny/16) * (ny/16)



/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {

    float* vnor = (float *)malloc(sizeof(float) * ny * nx); //normalized data
    vector<float> rs(ny, 0.0);  //sum per row
    vector<float> rss(ny, 0.0); //sum of square per row

    
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
    float* dGPU = NULL;  // data ny*nx matrix
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL; // result ny*ny matrix
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    // copy normalized matrix to dGPU
    CHECK(cudaMemcpy(dGPU, vnor, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);  // block size 16*16
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y)); // number of blocks
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    // rGPU to result
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

}
