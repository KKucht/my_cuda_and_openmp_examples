#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;

#define N 10

void read_data(unsigned char* in_image, unsigned char* out_image, long long int width, long long int height, long long int padded_width) {
    for (long long int i = 0; i < height ; i++) {
        memcpy(out_image + i * width, in_image + (i + 1) * padded_width + 1, width * sizeof(unsigned char));
    }
}

void prep_data(unsigned char* in_image, unsigned char* out_image, long long int width, long long int height, long long int padded_width) {
    for (int i = 0; i < height ; ++i) {
        memcpy(out_image + (i + 1) * padded_width + 1, in_image + i * width, width * sizeof(unsigned char));
    }
}

__global__ void sobel_operator_fourth(unsigned char* in_image, unsigned char* out_image, long long int width, long long  int height) {
    long long int x = blockIdx.x * blockDim.x + threadIdx.x;
    long long int y = blockIdx.y * blockDim.y + threadIdx.y;
    // // TEST IF MAP WORKS

    // for (long long int local_x = x; local_x < width;  local_x += (blockDim.x - 2)){
    //     for (long long int local_y = y; local_y < height;  local_y += (blockDim.y  - 2)){
    //         if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < blockDim.x && threadIdx.y < blockDim.y ){
    //             long long int idx_out = local_y * width + local_x;
    //             out_image[idx_out] = in_image[idx_out];
    //         }
    //     }
    // }

    int shared_x = threadIdx.x;
    int shared_y = threadIdx.y;

    __shared__ unsigned char shared_mem[32][32];

    for (long long int local_x = x; local_x < width ;  local_x += (blockDim.x - 2)* gridDim.x){
        for (long long int local_y = y; local_y < height ;  local_y += (blockDim.y - 2)* gridDim.x){

            shared_mem[shared_x][shared_y] = in_image[local_y * width + local_x];

            __syncthreads();
            if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < blockDim.x && threadIdx.y < blockDim.y ) {
                int sumx = 0;
                int sumy = 0;

                sumx -=      shared_mem[shared_x  - 1][shared_y - 1];
                sumx +=      shared_mem[shared_x  + 1][shared_y - 1];
                sumx -=  2 * shared_mem[shared_x  - 1][shared_y    ];
                sumx +=  2 * shared_mem[shared_x  + 1][shared_y    ];
                sumx -=      shared_mem[shared_x  - 1][shared_y + 1];
                sumx +=      shared_mem[shared_x  + 1][shared_y + 1];

                sumy -=      shared_mem[shared_x  - 1][shared_y - 1];
                sumy -=  2 * shared_mem[shared_x     ][shared_y - 1];
                sumy -=      shared_mem[shared_x  + 1][shared_y - 1];
                sumy +=      shared_mem[shared_x  - 1][shared_y + 1];
                sumy +=  2 * shared_mem[shared_x     ][shared_y + 1];
                sumy +=      shared_mem[shared_x  + 1][shared_y + 1];

                int magnitude = sqrtf(sumx * sumx + sumy * sumy);

                long long int idx_out = local_y * width + local_x;
                out_image[idx_out] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
            }
            __syncthreads();


            // long long int idx_out = local_y * width + local_x;
            // out_image[idx_out] = shared_mem[shared_x][shared_y];
        }
    }

}

__global__ void sobel_operator_third(unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
    long long int x = blockIdx.x * blockDim.x + threadIdx.x;
    long long int y = blockIdx.y * blockDim.y + threadIdx.y;

    long long local_x = x + 1;
    long long local_y = y + 1;

    if (local_x < width - 1 && local_y < height - 1) {

        int sumx = 0;
        int sumy = 0;

        sumx -=      in_image[(local_y - 1)* width + local_x  - 1];
        sumx +=      in_image[(local_y - 1)* width + local_x  + 1];
        sumx -=  2 * in_image[(local_y    )* width + local_x  - 1];
        sumx +=  2 * in_image[(local_y    )* width + local_x  + 1];
        sumx -=      in_image[(local_y + 1)* width + local_x  - 1];
        sumx +=      in_image[(local_y + 1)* width + local_x  + 1];

        sumy -=      in_image[(local_y - 1)* width + local_x  - 1];
        sumy -=  2 * in_image[(local_y - 1)* width + local_x     ];
        sumy -=      in_image[(local_y - 1)* width + local_x  + 1];
        sumy +=      in_image[(local_y + 1)* width + local_x  - 1];
        sumy +=  2 * in_image[(local_y + 1)* width + local_x     ];
        sumy +=      in_image[(local_y + 1)* width + local_x  + 1];

        int magnitude = sqrtf(sumx * sumx + sumy * sumy);

        long long idx_out = local_y * width + local_x;
        out_image[idx_out] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    
    }
}

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

__global__ void sobel_operator_second(unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
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
    printf("works.\n");
    if (argc == 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        size = rows * cols * sizeof(unsigned char);

        image_array1 = (unsigned char *)malloc(size);

        if (image_array1 == NULL) {
            printf("Nie udało się zaalokować pamięci image_array1.\n");
            return -1;
        }

        image_array2 = (unsigned char *)malloc(size);
    
        if (image_array2 == NULL) {
            printf("Nie udało się zaalokować pamięci image_array2.\n");
            return -1;
        }

        printf("generate.\n");
        for (long long i = 0; i < size; i++) {
            image_array1[i] = rand() % 256;  // Random grayscale image
        }
        printf("finished.\n");
    }
    else {

        cv::Mat image = cv::imread("imgin.png", cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            printf("Nie można otworzyć lub znaleźć obrazu.\n");
            return -1;
        }
        rows = image.rows;
        cols = image.cols;
        size = rows * cols * sizeof(unsigned char);

        image_array1 = (unsigned char *)malloc(size);
        image_array2 = (unsigned char *)malloc(size);
    
        if (image_array1 == NULL || image_array2 == NULL) {
            printf("Nie udało się zaalokować pamięci.\n");
            return -1;
        }

        for (int i = 0; i < rows; ++i) {
            memcpy(image_array1 + i * cols, image.ptr(i), cols * sizeof(unsigned char));
        }


    }

    printf("There will be avarege time for N = %d.\n", N);
    

    // load image

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
        sobel_operator_second<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, cols, rows);
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

    start_time1 = clock();
    for (int i = 0 ; i< N ; i++ ) {
        start_timers1[i] = clock();
        sobel_operator_third<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, cols, rows);
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
    printf("Third: Sobel operator completed in %.10f miliseconds\n", elapsed_time);

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
    // printf("0");
    // fflush(stdout);

    cudaMemcpy(image_array2, d_output, size, cudaMemcpyDeviceToHost);

    // cudaFree(d_input);
    // cudaFree(d_output);



    // /// PREPERE, to NEW BLOCKS ADDITIONAL
    // long long int real_nr_of_blocks_x = (cols + threadsPerBlock.x - 1 - 2) / (threadsPerBlock.x - 2);
    // long long int real_nr_of_blocks_y = (rows + threadsPerBlock.y - 1 - 2) / (threadsPerBlock.y - 2);

    // real_nr_of_blocks_x = real_nr_of_blocks_x > blocksPerGrid.x ? real_nr_of_blocks_x : blocksPerGrid.x;
    // real_nr_of_blocks_y = real_nr_of_blocks_y > blocksPerGrid.y ? real_nr_of_blocks_y : blocksPerGrid.y;

    // long long int padded_width = real_nr_of_blocks_x * (threadsPerBlock.x - 2 ) + 2;
    // long long int padded_height = real_nr_of_blocks_y * (threadsPerBlock.y - 2) + 2;
    // size = padded_width * padded_height ;

    // unsigned char * new_in_image = (unsigned char *)calloc(size , sizeof(unsigned char));
    // unsigned char * new_out_image = (unsigned char *)calloc(size , sizeof(unsigned char));

    // size *= sizeof(unsigned char);

    
    // // printf("1");
    // // fflush(stdout);

    // prep_data (image_array1, new_in_image, cols, rows, padded_width);

    // // printf("2");
    // // fflush(stdout);

    // // printf("3");
    // // fflush(stdout);

    // cudaMalloc((void **)&d_input, size);
    // cudaMalloc((void **)&d_output, size);
    // checkCudaError("Memory allocation");

    // // printf("\n%lld\n", size);
    // // printf("4");
    // // fflush(stdout);

    // cudaMemcpy(d_input, new_in_image, size, cudaMemcpyHostToDevice);
    // checkCudaError("Memory copy to device");

    // // printf("5");
    // // fflush(stdout);

    // start_time1 = clock();
    // for (int i = 0 ; i< N ; i++ ) {
    //     sobel_operator_fourth<<<threadsPerBlock, blocksPerGrid>>>(d_input, d_output, padded_width, padded_height);
    //     cudaDeviceSynchronize();
    //     checkCudaError("Kernel execution");
    // }
    // end_time1 = clock();

    // start_time2 = clock();
    // for (int i = 0 ; i< N ; i++ ) {
    //     sobel_operator_empty<<<threadsPerBlock, blocksPerGrid>>>(d_input, d_output, padded_width, padded_height);
    //     cudaDeviceSynchronize();
    //     checkCudaError("Kernel execution");
    // }
    // end_time2 = clock();

    // elapsed_time = (double)((end_time1 - start_time1) - (end_time2 - start_time2)) / (CLOCKS_PER_SEC/1000) / N;
    // printf("Fourth: Sobel operator completed in %.10f miliseconds\n", elapsed_time);

    // cudaMemcpy(new_out_image, d_output, size, cudaMemcpyDeviceToHost);
    // checkCudaError("Memory copy to host");

    // read_data(new_out_image, image_array2, cols, rows, padded_width);

    // free (new_out_image);

    // save image
    printf("Generate image\n");
    cv::Mat new_image(rows, cols, CV_8UC1);
    for (long long i = 0; i < rows; ++i) {
        memcpy(new_image.ptr(i), image_array2 + i * cols, cols * sizeof(unsigned char));
    }

    cv::imwrite("imgout.png", new_image);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(image_array1);
    free(image_array2);



    return 0;
}
