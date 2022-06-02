#include <vector>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

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


__global__ void mykernel(int ny, int nx, const float* vda, int* y0v, int* x0v, int* y1v, int* x1v,  float* bestv ) {
    int wx = threadIdx.x + blockIdx.x * blockDim.x;
    int wy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (wy == 0 || wx == 0 || wx > nx || wy > ny){
        return;
    }
    
    float psize = ny*nx;
    float sum_all = vda[nx + (nx+1) * ny];
    float wsize = wy*wx;
    float wsize1 = 1 / wsize;
    float bsize1 = 1 / (psize-wsize);

    float util_best = -1;
    int x0_best = 0; 
    int y0_best = 0;
    int x1_best = 0;
    int y1_best = 0;
    
    for (int y0 = 0; y0 <= ny - wy; y0++){
        for (int x0 = 0; x0 <= nx - wx; x0++){
            int y1=y0+wy;
            int x1=x0+wx;
            float sumin = vda[x1 + (nx+1) * y1] - vda[x0 + (nx+1) * y1]
                           + vda[x0 + (nx+1) * y0] - vda[x1 + (nx+1) * y0];
            float sumout = sum_all - sumin;
            float util = sumin*sumin*wsize1 + sumout*sumout*bsize1;
            if (util > util_best)
            {
                util_best = util;
                y0_best = y0;
                x0_best = x0;
                y1_best = y1;
                x1_best = x1;
            }
        }
    }

    bestv[(wx-1) + nx* (wy-1)] = util_best;
    y0v[(wx-1) + nx * (wy-1)] = y0_best;
    x0v[(wx-1) + nx * (wy-1)] = x0_best;
    y1v[(wx-1) + nx * (wy-1)] = y1_best;
    x1v[(wx-1) + nx * (wy-1)] = x1_best;
    
}


Result segment(int ny, int nx, const float* data) {
    // precomputing
    vector<float> vda((nx+1) * (ny+1));
    for(int y=1; y<=ny; y++){
        for(int x=1; x<=nx; x++){
                vda[x+y*(nx+1)]=data[3*(x-1)+3*nx*(y-1)]+vda[(x-1)+y*(nx+1)];
        }
    }
    for(int x=1; x<=nx; x++){
        for(int y=1; y<=ny; y++){
            vda[x+y*(nx+1)] += vda[x+(y-1)*(nx+1)];
        }
    }
    
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, (nx+1) * (ny+1) * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, vda.data(), (nx+1) * (ny+1) * sizeof(float), cudaMemcpyHostToDevice));
    int* y0GPU = NULL;
    int* x0GPU = NULL;
    int* y1GPU = NULL;
    int* x1GPU = NULL;
    float* bestGPU = NULL;
    CHECK(cudaMalloc((void**)&y0GPU, nx * ny * sizeof(int)));
    CHECK(cudaMalloc((void**)&x0GPU, nx * ny * sizeof(int)));
    CHECK(cudaMalloc((void**)&y1GPU, nx * ny * sizeof(int)));
    CHECK(cudaMalloc((void**)&x1GPU, nx * ny * sizeof(int)));
    CHECK(cudaMalloc((void**)&bestGPU, nx * ny * sizeof(float)));
    
    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, y0GPU, x0GPU, y1GPU, x1GPU, bestGPU);
    CHECK(cudaGetLastError());

    
    vector<int> y0v(ny * nx);
    vector<int> x0v(ny * nx);
    vector<int> y1v(ny * nx);
    vector<int> x1v(ny * nx);
    vector<float> bestv(ny * nx);
    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(y0v.data(), y0GPU, ny * nx * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(x0v.data(), x0GPU, ny * nx * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y1v.data(), y1GPU, ny * nx * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(x1v.data(), x1GPU, ny * nx * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bestv.data(), bestGPU, ny * nx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(bestGPU));
    CHECK(cudaFree(y0GPU));
    CHECK(cudaFree(x0GPU));
    CHECK(cudaFree(y1GPU)); 
    CHECK(cudaFree(x1GPU));
    

    // find largest utility
    int k = 0;
    float tmp = -1;
    for(int i=0; i<ny*nx; i++){
        if(bestv[i] > tmp){
            tmp = bestv[i];
            k = i;
        }
    }
    int r_y0 = y0v[k];
    int r_x0 = x0v[k];
    int r_y1 = y1v[k];
    int r_x1 = x1v[k];  
	
    float psize = nx*ny;
    float wsize = (r_y1-r_y0)*(r_x1-r_x0) ;
    float bsize = psize - wsize;
    float sum_all = vda[nx + (nx+1) * ny];
    float sumin = vda[r_x1 + (nx+1) * r_y1] - vda[r_x0 + (nx+1) * r_y1] 
                    - vda[r_x1 + (nx+1) * r_y0] + vda[r_x0 + (nx+1) * r_y0];
    float sumout = sum_all - sumin;
    float outer = sumout / bsize;
    float inner = sumin / wsize;
    Result result {
	    r_y0, r_x0, r_y1, r_x1, 
	    {outer, outer, outer}, 
	    {inner, inner, inner} 
    };
    return result;
}
