#include <algorithm>
#include <omp.h>
#include <iostream>

using namespace std;

typedef unsigned long long data_t;

//round down to next power of two
int lastPow2 (int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x/2;
}

void psort(int n, data_t* data){
    int max_thread = omp_get_max_threads();

    // number of threads created
    // each thread for one block
    int num_thread = lastPow2(max_thread); 


    if (num_thread < 2)
    {
        sort(data, data+n);
        return;
    }
    
    int block_size = n / num_thread;
    if(n % num_thread) {
        block_size+=1;
    }

    #pragma omp parallel for schedule(static, 1)
    for(int i=1; i<=num_thread; i++){
        int thread_num = omp_get_thread_num();
        int start = min(thread_num*block_size, n);
        int end = min((thread_num+1)*block_size, n);
        sort(data+start, data+end);
    }


    // merge pairs of blocks
    num_thread /= 2;

    while(num_thread > 0){   
        #pragma omp parallel for schedule(static, 1)
        for(int i=0; i<num_thread; i++)
        {
            int thread_i = omp_get_thread_num()*2;

            int first = thread_i * block_size;
            int middle = min((thread_i+1) * block_size, n);
            int end = min((thread_i+2) * block_size, n);

            inplace_merge(data+first, data+middle, data+end);
        }
    block_size *= 2;
    num_thread /= 2;
    }
}
