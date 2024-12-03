# to create generator
nvcc -o generate generate.cu raw_image_reader.cpp `pkg-config --cflags --libs opencv4`

# usage of generator
./generate or ./generate number number

# to start testing my sobel operators
./run.sh
