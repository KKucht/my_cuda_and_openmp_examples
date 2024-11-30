#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "raw_image_reader.hpp"

__device__ unsigned char random_number(long long seed, long long x, long long y) {
    long long hash = seed ^ (x * 31 + y * 71);
    hash = (hash ^ (hash >> 21)) * 2654435761;
    hash = (hash ^ (hash >> 13)) * 2654435761;
    return (unsigned char)((hash ^ (hash >> 16)) % 256);
}

__global__ void generate( unsigned char* image, long long width, long long height) {
    long long int x = blockIdx.x * blockDim.x + threadIdx.x;
    long long int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - 1 && y < height - 1)
        image[y* width + x] = random_number(x*y + 1231312344, x, y);
}

void checkCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc, char **argv) {
    unsigned char *image_array1 = nullptr;
    long long rows = 0;
    long long cols = 0;
    long long size = 0;
    cv::Mat image;

    if (argc == 3) {   
        rows = std::atoll(argv[1]);
        cols = std::atoll(argv[2]);
        size = rows * cols;

        image_array1 = (unsigned char *)malloc(size * sizeof(unsigned char));
        
        if (image_array1 == nullptr) {
            printf("Nie udało się zaalokować pamięci image_array1.\n");
            return -1;
        }

        unsigned char *d_image;
        cudaMalloc((void **)&d_image, size);
        checkCudaError("Memory allocation");

        dim3 threadsPerBlock(32, 32);

        dim3 blocksPerGrid(
            (cols + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1 ) / threadsPerBlock.y
        );

        printf("Generowanie obrazu...\n");
        generate<<<blocksPerGrid, threadsPerBlock>>>(d_image, cols, rows);
        cudaDeviceSynchronize();
        checkCudaError("Kernel execution");

        cudaMemcpy(image_array1, d_image, size, cudaMemcpyDeviceToHost);
        printf("Generowanie zakończone.\n");

        // image = cv::Mat(rows, cols, CV_8UC1, image_array1).clone();
        // for (unsigned long long i = 0; i < rows; ++i) {
        //     std::memcpy(image.ptr(i), image_array1 + i * cols, cols * sizeof(unsigned char));
        // }
    } else {
        image = cv::imread("img.png", cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            printf("Nie można otworzyć lub znaleźć obrazu.\n");
            return -1;
        }
        rows = image.rows;
        cols = image.cols;
        size = rows * cols * sizeof(unsigned char);

        image_array1 = (unsigned char *)malloc(size);

        for (unsigned long long i = 0; i < rows; ++i) {
            memcpy(image_array1 + i * cols, image.ptr(i), cols * sizeof(unsigned char));
        }
    }


    printf("Zapisywanie obrazu...\n");
    raw::writeImageRAW("imgin", image_array1, cols, rows);
    // if (!cv::imwrite("imgin.png", image)) {
    //     printf("Błąd podczas zapisywania obrazu.\n");
    // }

    if (image_array1 != nullptr) {
        free(image_array1);
    }

    return 0;
}