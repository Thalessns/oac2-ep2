// convolucao2d.cu - Versão CUDA da convolução 2D
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ==================== CONSTANTES ====================
const int KERNEL_SIZE = 3;
__constant__ float DEVICE_KERNEL[9];  // Memória constante da GPU

// ==================== ESTRUTURA IMAGEM ====================
struct Image {
    int width;
    int height;
    std::vector<float> data;
    
    Image(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}
    
    float& operator()(int y, int x) { return data[y * width + x]; }
    const float& operator()(int y, int x) const { return data[y * width + x]; }
};

// ==================== KERNEL CUDA ====================
__global__ void convolucaoCUDAKernel(const float* input, float* output, 
                                     int width, int height) {
    // Calcular coordenadas do pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Verificar limites
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int kernelRadius = 1;
    
    // Convolução 3x3
    for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int iy = y + ky;
            int ix = x + kx;
            
            // Tratamento de bordas (clamp)
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            
            // Índice no kernel
            int kernelIdx = (ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius);
            
            // Acessar pixel de entrada e multiplicar pelo kernel
            sum += input[iy * width + ix] * DEVICE_KERNEL[kernelIdx];
        }
    }
    
    // Escrever resultado
    output[y * width + x] = sum;
}

// ==================== KERNEL COM SHARED MEMORY (otimizado) ====================
__global__ void convolucaoCUDASharedKernel(const float* input, float* output, 
                                          int width, int height) {
    // Shared memory para tile da imagem
    __shared__ float tile[34][34];  // 32x32 + bordas de 1 pixel cada lado
    
    // Coordenadas globais
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Coordenadas locais no tile
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Carregar pixel central do tile
    if (gx < width && gy < height) {
        tile[ly + 1][lx + 1] = input[gy * width + gx];
    } else {
        tile[ly + 1][lx + 1] = 0.0f;
    }
    
    // Carregar bordas do tile (requer cooperação entre threads)
    // Borda esquerda
    if (lx == 0 && gx > 0) {
        int border_x = gx - 1;
        if (gy < height) {
            tile[ly + 1][0] = input[gy * width + border_x];
        }
    }
    
    // Borda direita
    if (lx == blockDim.x - 1 && gx < width - 1) {
        int border_x = gx + 1;
        if (gy < height) {
            tile[ly + 1][blockDim.x + 1] = input[gy * width + border_x];
        }
    }
    
    // Borda superior
    if (ly == 0 && gy > 0) {
        int border_y = gy - 1;
        if (gx < width) {
            tile[0][lx + 1] = input[border_y * width + gx];
        }
    }
    
    // Borda inferior
    if (ly == blockDim.y - 1 && gy < height - 1) {
        int border_y = gy + 1;
        if (gx < width) {
            tile[blockDim.y + 1][lx + 1] = input[border_y * width + gx];
        }
    }
    
    // Sincronizar todas as threads do bloco
    __syncthreads();
    
    // Aplicar convolução apenas para threads que processam pixels válidos
    if (gx < width && gy < height) {
        float sum = 0.0f;
        
        // Convolução usando shared memory
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int kernelIdx = ky * 3 + kx;
                sum += tile[ly + ky][lx + kx] * DEVICE_KERNEL[kernelIdx];
            }
        }
        
        output[gy * width + gx] = sum;
    }
}

// ==================== FUNÇÕES AUXILIARES ====================
Image gerarImagemTeste(int width, int height) {
    Image img(width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Padrão de teste (gradiente)
            img(i, j) = (i * 0.3f + j * 0.7f);
        }
    }
    return img;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Erro CUDA (" << msg << "): " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ==================== VERSÃO CUDA ====================
double convolucaoCUDA(const Image& input, bool useSharedMemory = false) {
    // Preparar kernel na CPU
    float hostKernel[9];
    for (int i = 0; i < 9; i++) {
        hostKernel[i] = 1.0f / 9.0f;
    }
    
    // Copiar kernel para memória constante da GPU
    checkCudaError(
        cudaMemcpyToSymbol(DEVICE_KERNEL, hostKernel, 9 * sizeof(float)),
        "copia kernel para constante"
    );
    
    // Alocar memória na GPU
    size_t size = input.width * input.height * sizeof(float);
    float *d_input, *d_output;
    
    checkCudaError(cudaMalloc(&d_input, size), "alocar input GPU");
    checkCudaError(cudaMalloc(&d_output, size), "alocar output GPU");
    
    // Copiar imagem de entrada para GPU
    checkCudaError(
        cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice),
        "copia input para GPU"
    );
    
    // Configurar grid e blocos
    dim3 blockSize(32, 32);  // 32x32 = 1024 threads por bloco (máximo)
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Sincronizar e criar eventos para medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Medir tempo de execução do kernel
    cudaEventRecord(start);
    
    if (useSharedMemory) {
        // Executar kernel com shared memory
        convolucaoCUDASharedKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                           input.width, input.height);
    } else {
        // Executar kernel simples
        convolucaoCUDAKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                     input.width, input.height);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Verificar erros do kernel
    checkCudaError(cudaGetLastError(), "execucao kernel");
    
    // Calcular tempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Criar imagem de saída
    Image output(input.width, input.height);
    
    // Copiar resultado de volta para CPU
    checkCudaError(
        cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost),
        "copia output para CPU"
    );
    
    // Liberar memória GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 1000.0;  // Converter para segundos
}

// ==================== VERSÃO SEQUENCIAL (para comparação) ====================
Image convolucaoSequencial(const Image& input) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    float kernel[9];
    for (int i = 0; i < 9; i++) kernel[i] = 1.0f / 9.0f;
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                    int iy = y + ky;
                    int ix = x + kx;
                    
                    if (iy < 0) iy = 0;
                    if (iy >= input.height) iy = input.height - 1;
                    if (ix < 0) ix = 0;
                    if (ix >= input.width) ix = input.width - 1;
                    
                    int kernelIdx = (ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius);
                    sum += input(iy, ix) * kernel[kernelIdx];
                }
            }
            output(y, x) = sum;
        }
    }
    
    return output;
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    // Inicializar CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "=== EXPERIMENTO DE CONVOLUÇÃO 2D (CUDA) ===\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memória Global: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Máx. Threads por Bloco: " << prop.maxThreadsPerBlock << "\n\n";
    
    // Resoluções a testar
    std::vector<std::pair<int, int>> resolucoes = {
        {512, 512},
        {1024, 1024},
        {4096, 4096}
    };
    
    for (const auto& res : resolucoes) {
        int width = res.first;
        int height = res.second;
        
        std::cout << "Resolução: " << width << "x" << height << "\n";
        std::cout << "Gerando imagem de teste (" << width * height * sizeof(float) / (1024*1024) 
                  << " MB)...\n";
        
        Image img = gerarImagemTeste(width, height);
        
        // Versão sequencial (CPU)
        std::cout << "Executando versão sequencial (CPU)...\n";
        auto inicio = std::chrono::high_resolution_clock::now();
        Image out_seq = convolucaoSequencial(img);
        auto fim = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> tempo_seq = fim - inicio;
        std::cout << "Tempo CPU: " << tempo_seq.count() << "s\n";
        
        // Versão CUDA simples
        std::cout << "Executando versão CUDA (sem shared memory)...\n";
        double tempo_cuda_simple = convolucaoCUDA(img, false);
        std::cout << "Tempo CUDA (simples): " << tempo_cuda_simple << "s\n";
        std::cout << "Speedup: " << tempo_seq.count() / tempo_cuda_simple << "x\n";
        
        // Versão CUDA com shared memory
        std::cout << "Executando versão CUDA (com shared memory)...\n";
        double tempo_cuda_shared = convolucaoCUDA(img, true);
        std::cout << "Tempo CUDA (shared): " << tempo_cuda_shared << "s\n";
        std::cout << "Speedup: " << tempo_seq.count() / tempo_cuda_shared << "x\n";
        
        // Comparação shared vs simple
        std::cout << "Ganho shared vs simple: " 
                  << tempo_cuda_simple / tempo_cuda_shared << "x\n";
        
        std::cout << "----------------------------------------\n\n";
    }
    
    // Finalizar CUDA
    cudaDeviceReset();
    
    return 0;
}