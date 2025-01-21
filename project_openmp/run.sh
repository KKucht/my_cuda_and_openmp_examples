#!/bin/bash

# Kompilacja programów OpenMP
g++ -fopenmp -o sobel_v1 sobel_v1.cpp raw_image_reader.cpp `pkg-config --cflags --libs opencv4`
g++ -fopenmp -o sobel_v2 sobel_v2.cpp raw_image_reader.cpp `pkg-config --cflags --libs opencv4`
g++ -fopenmp -o sobel_v3 sobel_v3.cpp raw_image_reader.cpp `pkg-config --cflags --libs opencv4`
g++ -fopenmp -o sobel_v4 sobel_v4.cpp raw_image_reader.cpp `pkg-config --cflags --libs opencv4`

./sobel_v1
./sobel_v2
./sobel_v3
./sobel_v4
