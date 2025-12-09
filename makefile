# Makefile para compilar
NVCC = nvcc
CFLAGS = -std=c++11 -O2
NVCC_FLAGS = $(CFLAGS) -Xcompiler -fno-strict-aliasing

# Alvo principal
all: cuda

# Versão CUDA (requer CUDA Toolkit)
cuda: convolucao2d.cu
	$(NVCC) $(NVCC_FLAGS) convolucao2d.cu -o convolucao_cuda

# Limpar arquivos compilados
clean:
	rm -f convolucao_cuda *.pgm
