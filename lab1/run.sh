#!/bin/bash

# Kompilacja programów CUDA
nvcc -o raw raw.cu
nvcc -o shared shared.cu
nvcc -o shared_basic_al shared_basic_al.cu
nvcc -o shared_basic_al_more_blocks shared_basic_al_more_blocks.cu

# Ustawienie wartości zmiennych
N=100000000
a=1
b=100

# Uruchomienie programu raw
echo "Running ./raw with:"
echo "N=$N, a=$a, b=$b"
echo -e "$N\n$a\n$b\n" | ./raw

# Uruchomienie programu shared
echo "Running ./shared with:"
echo "N=$N, a=$a, b=$b"
echo -e "$N\n$a\n$b\n" | ./shared

# Uruchomienie programu shared_basic_al
echo "Running ./shared_basic_al with:"
echo "N=$N, a=$a, b=$b"
echo -e "$N\n$a\n$b" | ./shared_basic_al

echo "Running ./shared_basic_al_more_blocks with:"
echo "N=$N, a=$a, b=$b"
echo -e "$N\n$a\n$b" | ./shared_basic_al_more_blocks
