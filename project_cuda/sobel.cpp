#include <stdio.h>
#include <opencv2/opencv.hpp>


int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

void sobel_operator(unsigned char* in_image, unsigned char* out_image, int width, int height) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            long long sumx = 0;
            long long sumy = 0;
            for (int p = -1; p <= 1; p++) {
                for (int q = -1; q <= 1; q++) {
                    int idx = (i + p) * width + (j + q);
                    sumx += (in_image[idx] * Gx[p + 1][q + 1]);
                    sumy += (in_image[idx] * Gy[p + 1][q + 1]);
                }
            }
            int magnitude = (int)sqrt(sumx * sumx + sumy * sumy);

            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            int idx_out = i * width + j;
            out_image[idx_out] = (unsigned char)magnitude;
        }
    }
}



int main(int argc, char **argv) {
    cv::Mat image = cv::imread("imgin.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Nie można otworzyć lub znaleźć obrazu.\n");
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;

    unsigned char *image_array1 = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
    unsigned char *image_array2 = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
    if (image_array1 == NULL || image_array2 == NULL) {
        printf("Nie udało się zaalokować pamięci.\n");
        return -1;
    }

    for (int i = 0; i < rows; ++i) {
        memcpy(image_array1 + i * cols, image.ptr(i), cols * sizeof(unsigned char));
    }
    // load image

    // Sobel operator
    sobel_operator(image_array1, image_array2, cols, rows);

    // save image
    cv::Mat new_image(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        memcpy(new_image.ptr(i), image_array2 + i * cols, cols * sizeof(unsigned char));
    }

    cv::imwrite("imgout.png", new_image);

    free(image_array1);
    free(image_array2);

    printf("Sobel operator completed.\n");

    return 0;
}

