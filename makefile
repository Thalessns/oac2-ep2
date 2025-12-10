# Makefile para compilar
NVCC = nvcc
CC = g++
CFLAGS = -std=c++11 -O2
NVCC_FLAGS = $(CFLAGS) -Xcompiler -fno-strict-aliasing
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Alvo principal
all: cpu cuda

# Versão cpu
cpu: convolucao2d.cpp
	$(CC) $(CFLAGS) -fopenmp -o convolucao_cpu convolucao2d.cpp $(OPENCV_FLAGS)

# Versão CUDA (requer CUDA Toolkit)
cuda: convolucao2d.cu
	$(NVCC) $(NVCC_FLAGS) convolucao2d.cu -o convolucao_cuda

# Limpar arquivos compilados
clean:
	rm -f convolucao_cuda *.pgm
