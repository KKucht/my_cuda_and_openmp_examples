#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "raw_image_reader.hpp"

using namespace cv;

#define N 10

__constant__ int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__global__ void sobel_operator(unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
    long long int x = blockIdx.x * blockDim.x + threadIdx.x;
    long long int y = blockIdx.y * blockDim.y + threadIdx.y;

    long long local_x = x + 1;
    long long local_y = y + 1;

    if (local_x < width - 1 && local_y < height - 1) {
            
        int sumx = 0;
        int sumy = 0;
        for (int p = -1; p <= 1; p++) {
            for (int q = -1; q <= 1; q++) {
                long long idx = (local_y + p) * width + (local_x + q);
                sumx += (in_image[idx] * Gx[p + 1][q + 1]);
                sumy += (in_image[idx] * Gy[p + 1][q + 1]);
            }
        }

        int magnitude = sqrtf(sumx * sumx + sumy * sumy);

        long long idx_out = local_y * width + local_x;
        out_image[idx_out] = (unsigned char)(magnitude > 255 ? 255 : magnitude);

    }
}

__global__ void sobel_operator_empty(unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
}

void checkCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc, char **argv) {
    
    unsigned char *image_array1;
    unsigned char *image_array2;

    long long rows = 0;
    long long cols = 0;
    long long size = 0;
    double max = 0, min=0;

    if (!raw::readImageRAW("imgin", image_array1, cols, rows)) {
        printf("Nie można otworzyć lub znaleźć obrazu.\n");
        return -1;
    }

    printf("rows: %lld\ncols: %lld\n", rows, cols);
    size = rows * cols * sizeof(unsigned char);
    image_array2 = (unsigned char *)malloc(size);

    if (image_array1 == NULL || image_array2 == NULL) {
        printf("Nie udało się zaalokować pamięci.\n");
        return -1;
    }

    printf("There will be avarege time for N = %d.\n", N);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    checkCudaError("Memory allocation");

    cudaMemcpy(d_input, image_array1, size, cudaMemcpyHostToDevice);
    checkCudaError("Memory copy to device");
    // init clocks
    double elapsed_time;
    clock_t start_time1, end_time1, start_time2, end_time2;
    clock_t start_timers1[N], end_timers1[N], start_timers2[N], end_timers2[N];

    dim3 threadsPerBlock(32, 32);

    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1 ) / threadsPerBlock.y
    );

    start_time1 = clock();
    for (int i = 0 ; i< N ; i++ ) {
        start_timers1[i] = clock();
        sobel_operator<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, cols, rows);
        cudaDeviceSynchronize();
        checkCudaError("Kernel execution");
        end_timers1[i] = clock();
    }
    end_time1 = clock();

    start_time2 = clock();
    for (int i = 0 ; i< N ; i++ ) {
        start_timers2[i] = clock();
        sobel_operator_empty<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, cols, rows);
        cudaDeviceSynchronize();
        checkCudaError("Kernel execution");
        end_timers2[i] = clock();
    }
    end_time2 = clock();

    elapsed_time = (double)((end_time1 - start_time1) - (end_time2 - start_time2)) / (CLOCKS_PER_SEC/1000) / N;
    printf("Second: Sobel operator completed in %.10f miliseconds\n", elapsed_time);

    max = -99;
    min = 9999999;
    for (int i = 0; i < N; i++){
        elapsed_time = (double)((end_timers1[i] - start_timers1[i]) - (end_timers2[i] - start_timers2[i])) / (CLOCKS_PER_SEC/1000) / N;
        if (min > elapsed_time){
            min = elapsed_time;
        }
        if (max < elapsed_time){
            max = elapsed_time;
        }
    }

    printf("Uncertainty: %.10f\n",  (max - min) / 2.0);

    cudaMemcpy(image_array2, d_output, size, cudaMemcpyDeviceToHost);

    printf("Generate image\n");
    cv::Mat new_image(rows, cols, CV_8UC1);
    for (unsigned long long int i = 0; i < rows; ++i) {
        memcpy(new_image.ptr(i), image_array2 + i * cols, cols * sizeof(unsigned char));
    }

    cv::imwrite("imgout.png", new_image);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(image_array1);
    free(image_array2);

    return 0;
}