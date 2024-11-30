#!/bin/bash

# Kompilacja programów CUDA
nvcc -o sobel_v1 sobel_v1.cu raw_image_reader.cpp `pkg-config --cflags --libs opencv4`
nvcc -o sobel_v2 sobel_v2.cu raw_image_reader.cpp `pkg-config --cflags --libs opencv4`
nvcc -o sobel_v3 sobel_v3.cu raw_image_reader.cpp `pkg-config --cflags --libs opencv4`

./sobel_v1
./sobel_v2
./sobel_v3
