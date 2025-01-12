/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 with Unified Memory
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

//elements generation
__global__ 
void calculate(int *result, int nr) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    int check_nr = my_index + 2;
    if (check_nr < nr){
        if (nr % check_nr == 0)
            atomicAdd(result, 1);
    }

}


int main(int argc,char **argv) {

    int N;

    printf("Enter number of elements: \n");
    scanf("%d", &N);

    long long result;
    int threadsinblock=1024;
    int blocksingrid=(N + threadsinblock - 1 ) / threadsinblock;

    int size = 1;

    int *results;

    //unified memory allocation - available for host and device
    if (cudaSuccess!=cudaMallocManaged(&results,size*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    printf("%u, %u\n", blocksingrid,threadsinblock);

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(results, N);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //device synchronization to ensure that data in memory is ready
    cudaDeviceSynchronize();

    //calculate sum of all elements
    result=0;
    for(int i=0;i<size;i++) {
      result = result + results[i];
    }
    if (result == 0){
        printf("Liczba pierwsza");
    }
    else {
        printf("nie pierwsza");
    }

    printf("\nSum of all elements is  %lld\n",result);

    //free memory
    if (cudaSuccess!=cudaFree(results))
      errorexit("Error when deallocating space on the GPU");

}
