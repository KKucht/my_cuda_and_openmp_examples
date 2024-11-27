/*
CUDA - prepare the histogram of N numbers in range of <a;b> where a and b should be integers
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeHistogram(unsigned long long *data, int N, unsigned long long *result) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int localIdx = threadIdx.x;

    __shared__ unsigned long long sharedData[1024];
    
    int s = blockDim.x;
    sharedData[localIdx] = data[idx];
    sharedData[localIdx] += data[idx + s];
    __syncthreads();
    for (s >>= 1;s > 0 ; s >>= 1){
        if (localIdx < s) {
            sharedData[localIdx] += sharedData[localIdx + s];
        }
        __syncthreads();
    }
    
    if (localIdx == 0) {
        atomicAdd(result, sharedData[localIdx]);
    }
}

void generateRandomNumbers(unsigned long long int *arr, int N, int A, int B) {
    
	srand(time(NULL));

    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }

}

int main(int argc,char **argv) {

    int threadsinblock=1024;
    int blocksingrid;

    int N,A,B;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    printf("Enter number of elements: \n");
    scanf("%d", &N);

	printf("Enter A value (start range): \n");
    scanf("%d", &A);

    printf("Enter B value (end range): \n");
    scanf("%d", &B);

    blocksingrid = ceil((double)N/threadsinblock/2);

	unsigned long long int *randomNumbers = (unsigned long long int *)calloc(blocksingrid * 1024 * 2,sizeof(unsigned long long int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	generateRandomNumbers(randomNumbers, N,A,B);

	printf("The kernel will run with: %d blocks\n", blocksingrid);

	unsigned long long int *resultArrayHost, *resultArrayDevice, *randomNumbersDevice;

	resultArrayHost = (unsigned long long int *)malloc(sizeof(unsigned long long int)*blocksingrid);
    

	if (resultArrayHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }
    
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&randomNumbersDevice, blocksingrid * 1024 * 2 * sizeof(long long int));
    cudaMalloc((void **)&resultArrayDevice,  sizeof(long long int));
    
    cudaMemcpy(randomNumbersDevice, randomNumbers, blocksingrid * 1024 * 2 * sizeof(long long int), cudaMemcpyHostToDevice);
    

    // Initialize device histogram to 0
    cudaMemset(resultArrayDevice, 0, sizeof(long long int));

    computeHistogram<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, N, resultArrayDevice);


    // Copy the histogram result back to the host
    cudaMemcpy(resultArrayHost, resultArrayDevice, sizeof(long long int), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print execution time
    

    unsigned long long sum = *resultArrayHost;
    sum /= N;
    printf("average value: %lld\n", sum);

	printf("Kernel execution time: %.3f ms\n", milliseconds);
    // Free allocated memory
    free(randomNumbers);
    free(resultArrayHost);
    cudaFree(randomNumbersDevice);
    cudaFree(resultArrayDevice);

    return 0;

}
