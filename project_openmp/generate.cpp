#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "raw_image_reader.hpp"
#include <omp.h>

// Funkcja generująca losową liczbę
unsigned char random_number(long long seed, long long x, long long y) {
    long long hash = seed ^ (x * 31 + y * 71);
    hash = (hash ^ (hash >> 21)) * 2654435761;
    hash = (hash ^ (hash >> 13)) * 2654435761;
    return (unsigned char)((hash ^ (hash >> 16)) % 256);
}

// Funkcja generująca obraz
void generate(unsigned char* image, long long width, long long height) {
    #pragma omp parallel for collapse(2)
    for (long long y = 0; y < height; ++y) {
        for (long long x = 0; x < width; ++x) {
            image[y * width + x] = random_number(x * y + 1231312344, x, y);
        }
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

        printf("Generowanie obrazu...\n");
        generate(image_array1, cols, rows);
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