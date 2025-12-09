# Makefile para compilar todas as versões

CC = g++
CFLAGS = -std=c++11 -O2
OMP_FLAGS = -fopenmp
CUDA_FLAGS = -DCUDA_ENABLED -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

# Alvo principal
all: sequencial threads openmp cuda

# Versão sequencial
sequencial: convolucao2d.cpp
	$(CC) $(CFLAGS) -o convolucao_seq convolucao2d.cpp

# Versão com threads
threads: convolucao2d.cpp
	$(CC) $(CFLAGS) -pthread -o convolucao_threads convolucao2d.cpp

# Versão OpenMP
openmp: convolucao2d.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o convolucao_omp convolucao2d.cpp

# Versão CUDA (requer CUDA Toolkit)
cuda: convolucao2d.cu
	nvcc -std=c++11 -O2 -o convolucao_cuda convolucao2d.cu

# Limpar arquivos compilados
clean:
	rm -f convolucao_seq convolucao_threads convolucao_omp convolucao_cuda *.pgm

# Executar testes
test: all
	@echo "Testando versão sequencial..."
	./convolucao_seq
	@echo "\nTestando versão com threads (4 threads)..."
	./convolucao_threads
	@echo "\nTestando versão OpenMP..."
	./convolucao_omp
	@if [ -f ./convolucao_cuda ]; then \
		echo "\nTestando versão CUDA..."; \
		./convolucao_cuda; \
	fi

.PHONY: all clean test