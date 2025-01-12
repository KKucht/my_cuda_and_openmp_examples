/*
CUDA - dynamic parallelism sample
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void generateRandomNumbers(int *arr, int N, int A, int B) {
    
	srand(time(NULL));

    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }

}

__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

// __device__
// int partition(int *arr, int low, int high) {
//     int pivot = arr[high];

//     int i = low - 1;

//     for (int j = low; j <= high - 1; j++) {
//         if(arr[j] < pivot){
//             arr[i] = arr[i] + arr[j];
//             arr[j] = arr[i] - arr[j];
//             arr[i] = arr[i] - arr[j];
//         }
//     }
//     arr[i + 1] = arr[i + 1] + arr[high];
//     arr[high] = arr[i + 1] - arr[high];
//     arr[i + 1] = arr[i + 1] - arr[high];
//     return i + 1;
// }

__global__
void quickSort (int *arr, int low, int high){
    if (low < high) {
        int pivot_index = low + (high - low)/2;
        int pivot = arr[pivot_index];
        int temp = 0;

        temp = arr[pivot_index];
        arr[pivot_index] = arr[high];
        arr[high] = temp;
        
        int current_index = low;

        for (int i = low; i < high ; i++) {
            if(arr[i] < pivot){
                temp = arr[current_index];
                arr[current_index] = arr[i];
                arr[i] = temp;
                current_index++;
            }
        }
        temp = arr[current_index];
        arr[current_index] = arr[high];
        arr[high] = temp;

        int pi = current_index;//= partition(arr, low, high);      

        quickSort<<<1, 1>>>(arr, low, pi - 1);

        quickSort<<<1, 1>>>(arr, pi + 1, high);
        
    }
}

void checkCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc,char **argv) {
     	cudaEvent_t start, stop;
    float milliseconds = 0;
    int N;

    printf("Enter number of elements: \n");
    scanf("%d", &N);

	int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    int * resultArrayHost = (int *)malloc(N * sizeof(int));

	generateRandomNumbers(randomNumbers, N,1,100000);

    int *randomNumbersDevice;
    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));

    if (cudaSuccess!=cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice))
      errorexit("Error during kernel launch in stream first");
 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /// HERE START
    quickSort<<<1, 1>>>(randomNumbersDevice, 0, N - 1);
    cudaDeviceSynchronize();

    cudaMemcpy(resultArrayHost, randomNumbersDevice, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("WWOOW");
    
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    for (int i = 0; i < N; i++){
        printf("%u\n", resultArrayHost[i]);
    }

	//run kernel on GPU 
	printf("Dynamic parallelism example\n");

    free(randomNumbers);
    cudaFree(randomNumbersDevice);
}