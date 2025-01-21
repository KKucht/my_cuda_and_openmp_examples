#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "raw_image_reader.hpp"
#include <omp.h>
#include <cmath>

using namespace cv;

#define N 10

const int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

void sobel_operator(const unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
    #pragma omp parallel for collapse(2)
    for (long long y = 1; y < height - 1; ++y) {
        for (long long x = 1; x < width - 1; ++x) {
            int sumx = 0;
            int sumy = 0;
            for (int p = -1; p <= 1; ++p) {
                for (int q = -1; q <= 1; ++q) {
                    long long idx = (y + p) * width + (x + q);
                    sumx += (in_image[idx] * Gx[p + 1][q + 1]);
                    sumy += (in_image[idx] * Gy[p + 1][q + 1]);
                }
            }
            int magnitude = static_cast<int>(sqrt(sumx * sumx + sumy * sumy));
            long long idx_out = y * width + x;
            out_image[idx_out] = static_cast<unsigned char>(magnitude > 255 ? 255 : magnitude);
        }
    }
}

void empty_sobel_operator(const unsigned char* in_image, unsigned char* out_image, long long width, long long height) {
}

int main(int argc, char **argv) {
    omp_set_num_threads(24);
    unsigned char *image_array1 = nullptr;
    unsigned char *image_array2 = nullptr;

    long long rows = 0;
    long long cols = 0;
    long long size = 0;
    double max = 0, min = 0;

    if (!raw::readImageRAW("imgin", image_array1, cols, rows)) {
        printf("Nie można otworzyć lub znaleźć obrazu.\n");
        return -1;
    }

    printf("rows: %lld\ncols: %lld\n", rows, cols);
    size = rows * cols * sizeof(unsigned char);
    image_array2 = new unsigned char[size];

    if (image_array1 == nullptr || image_array2 == nullptr) {
        printf("Nie udało się zaalokować pamięci.\n");
        return -1;
    }

    printf("There will be average time for N = %d.\n", N);

    double elapsed_time;
    clock_t start_time1, end_time1, start_time2, end_time2;
    clock_t start_timers1[N], end_timers1[N], start_timers2[N], end_timers2[N];

    start_time1 = clock();
    for (int i = 0; i < N; ++i) {
        start_timers1[i] = clock();
        sobel_operator(image_array1, image_array2, cols, rows);
        end_timers1[i] = clock();
    }
    end_time1 = clock();

    start_time2 = clock();
    for (int i = 0; i < N; ++i) {
        start_timers2[i] = clock();
        // Empty function call for timing purposes
        #pragma omp parallel for collapse(2)
        for (long long y = 0; y < rows; ++y) {
            for (long long x = 0; x < cols; ++x) {
                // Do nothing
            }
        }
        end_timers2[i] = clock();
    }
    end_time2 = clock();

    elapsed_time = (double)((end_time1 - start_time1) - (end_time2 - start_time2)) / (CLOCKS_PER_SEC / 1000) / N;
    printf("Second: Sobel operator completed in %.10f milliseconds\n", elapsed_time);

    max = -99;
    min = 9999999;
    for (int i = 0; i < N; ++i) {
        elapsed_time = (double)((end_timers1[i] - start_timers1[i]) - (end_timers2[i] - start_timers2[i])) / (CLOCKS_PER_SEC / 1000) / N;
        if (min > elapsed_time) {
            min = elapsed_time;
        }
        if (max < elapsed_time) {
            max = elapsed_time;
        }
    }

    printf("Uncertainty: %.10f\n", (max - min) / 2.0);

    printf("Generate image\n");
    // cv::Mat new_image(rows, cols, CV_8UC1);
    // for (unsigned long long int i = 0; i < rows; ++i) {
    //     memcpy(new_image.ptr(i), image_array2 + i * cols, cols * sizeof(unsigned char));
    // }
    // cv::imwrite("imgout.png", new_image);

    delete[] image_array1;
    delete[] image_array2;

    return 0;
}