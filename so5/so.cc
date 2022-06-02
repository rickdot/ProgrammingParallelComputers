#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>
#include <iostream>


using namespace std;

typedef unsigned long long data_t;







void print(data_t *a, int n){
    for(int i=0; i<n; i++){
        cout<<*(a+i)<<' ';
    }
    cout<<endl;
}


// choose the median of 3 numbers
data_t chooseMedian(data_t* a, data_t* b, data_t* c){
    
    // checking for a
    if((*b < *a && *a < *c) || (*c < *a && *a < *b)) 
        return *a;
    // checking for b
    else if((*a < *b && *b < *c) || (*c < *b && *b < *a)) 
        return *b;
    else
        return *c;
}

void quicksort(data_t* start, data_t* end){
    
    if(start==end) 
        return;



    data_t pivot = chooseMedian(start, (start+(end-start)/2), end-1);
    data_t* middle1 = partition(start, end, 
        [pivot](const auto &em) { return em < pivot; });
    data_t* middle2 = partition(middle1, end, 
        [pivot](const auto &em) { return !(pivot < em); });

    #pragma omp task
    quicksort(start, middle1);
    #pragma omp task
    quicksort(middle2, end);

}

void psort(int n, data_t* data) {

    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    // std::sort(data, data + n);  
    
    // print(data, n); 
    
    // chooseRandom(data, data+n);
    

    int max_thread = omp_get_max_threads();

    // #pragma omp parallel num_threads(max_thread)
    // #pragma omp single

    #pragma omp parallel
    #pragma omp single nowait
    {
        quicksort(data, data+n);
    }
}
