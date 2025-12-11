// convolucao2d.cpp - Implementações de convolução 2D para comparação de desempenho

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// ==================== CONSTANTES E ESTRUTURAS ====================

const int KERNEL_SIZE = 3;
const float KERNEL[9] = {
    1.0f/9, 1.0f/9, 1.0f/9,
    1.0f/9, 1.0f/9, 1.0f/9,
    1.0f/9, 1.0f/9, 1.0f/9
};

struct Image {
    int width;
    int height;
    std::vector<float> data;
    
    Image(int w = 0, int h = 0) : width(w), height(h), data(w * h, 0.0f) {}
    
    float& operator()(int y, int x) { return data[y * width + x]; }
    const float& operator()(int y, int x) const { return data[y * width + x]; }
    
    // Método para converter cv::Mat para Image (escala de cinza)
    static Image fromMat(const cv::Mat& mat) {
        if (mat.empty()) {
            return Image(0, 0);
        }
        
        Image img(mat.cols, mat.rows);
        
        if (mat.channels() == 1) {
            // Já está em escala de cinza
            for (int y = 0; y < mat.rows; y++) {
                for (int x = 0; x < mat.cols; x++) {
                    img(y, x) = mat.at<uchar>(y, x) / 255.0f;
                }
            }
        } else if (mat.channels() == 3) {
            // Converter BGR para escala de cinza
            for (int y = 0; y < mat.rows; y++) {
                for (int x = 0; x < mat.cols; x++) {
                    cv::Vec3b pixel = mat.at<cv::Vec3b>(y, x);
                    float gray = (pixel[0] * 0.114f +  // B
                                  pixel[1] * 0.587f +  // G
                                  pixel[2] * 0.299f);  // R
                    img(y, x) = gray / 255.0f;
                }
            }
        } else if (mat.channels() == 4) {
            // RGBA para escala de cinza (ignorar canal alpha)
            for (int y = 0; y < mat.rows; y++) {
                for (int x = 0; x < mat.cols; x++) {
                    cv::Vec4b pixel = mat.at<cv::Vec4b>(y, x);
                    float gray = (pixel[0] * 0.114f +  // B
                                  pixel[1] * 0.587f +  // G
                                  pixel[2] * 0.299f);  // R
                    img(y, x) = gray / 255.0f;
                }
            }
        }
        
        return img;
    }
    
    // Método para converter Image para cv::Mat
    cv::Mat toMat() const {
        cv::Mat mat(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float val = data[y * width + x];
                val = std::max(0.0f, std::min(1.0f, val));  // Clamp
                mat.at<uchar>(y, x) = static_cast<uchar>(val * 255);
            }
        }
        return mat;
    }
};

Image carregarImagem(const std::string& caminho, int redimensionar_largura = 0) {
    cv::Mat img = cv::imread(caminho, cv::IMREAD_COLOR);
    
    if (img.empty()) {
        throw std::runtime_error("Não foi possível carregar imagem: " + caminho);
    }
    
    // Redimensionar se especificado
    if (redimensionar_largura > 0 && redimensionar_largura != img.cols) {
        double escala = static_cast<double>(redimensionar_largura) / img.cols;
        int nova_altura = static_cast<int>(img.rows * escala);
        cv::resize(img, img, cv::Size(redimensionar_largura, nova_altura));
    }
    
    return Image::fromMat(img);
}

Image gerarImagemTeste(int width, int height) {
    Image img(width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Padrão de teste simples
            img(i, j) = (i + j) % 256;
        }
    }
    return img;
}

void salvarImagem(const Image& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;
    
    // Cabeçalho PGM simples
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    for (float val : img.data) {
        unsigned char pixel = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, val)));
        file.put(pixel);
    }
}


// ==================== VERSÃO 1: SEQUENCIAL ====================
Image convolucaoSequencial(const Image& input) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                    int iy = y + ky;
                    int ix = x + kx;
                    
                    // Tratamento de bordas: repetição de pixels
                    if (iy < 0) iy = 0;
                    if (iy >= input.height) iy = input.height - 1;
                    if (ix < 0) ix = 0;
                    if (ix >= input.width) ix = input.width - 1;
                    
                    int kernelIdx = (ky + kernelRadius) * KERNEL_SIZE + (kx + kernelRadius);
                    sum += input(iy, ix) * KERNEL[kernelIdx];
                }
            }
            output(y, x) = sum;
        }
    }
    
    return output;
}

// ==================== VERSÃO 2: THREADS EXPLÍCITAS (std::thread) ====================
Image convolucaoThreads(const Image& input, int numThreads) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    std::vector<std::thread> threads;
    
    // Função que cada thread executa
    auto processarLinhas = [&](int startRow, int endRow) {
        for (int y = startRow; y < endRow; y++) {
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
                        sum += input(iy, ix) * KERNEL[kernelIdx];
                    }
                }
                output(y, x) = sum;
            }
        }
    };
    
    // Distribuir linhas entre threads
    int rowsPerThread = input.height / numThreads;
    int startRow = 0;
    
    for (int i = 0; i < numThreads; i++) {
        int endRow = (i == numThreads - 1) ? input.height : startRow + rowsPerThread;
        threads.emplace_back(processarLinhas, startRow, endRow);
        startRow = endRow;
    }
    
    // Aguardar todas as threads
    for (auto& t : threads) {
        t.join();
    }
    
    return output;
}

// ==================== VERSÃO 3: OPENMP ====================
Image convolucaoOpenMPStatic(const Image& input, int scheduleType = 0) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    int maxThreads = omp_get_max_threads();

    #pragma omp parallel for num_threads(maxThreads) schedule(static)
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
                    sum += input(iy, ix) * KERNEL[kernelIdx];
                }
            }
            output(y, x) = sum;
        }
    }
    
    return output;
}

Image convolucaoOpenMPDynamic(const Image& input, int scheduleType = 0) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    int maxThreads = omp_get_max_threads();

    #pragma omp parallel for num_threads(maxThreads) schedule(dynamic)
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
                    sum += input(iy, ix) * KERNEL[kernelIdx];
                }
            }
            output(y, x) = sum;
        }
    }
    
    return output;
}

Image convolucaoOpenMPCollapse(const Image& input, int scheduleType = 0) {
    Image output(input.width, input.height);
    int kernelRadius = KERNEL_SIZE / 2;
    int maxThreads = omp_get_max_threads();

    #pragma omp parallel for num_threads(maxThreads) collapse(2)
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
                    sum += input(iy, ix) * KERNEL[kernelIdx];
                }
            }
            output(y, x) = sum;
        }
    }
    
    return output;
}

// ==================== VERSÃO 4: CUDA (implementação simplificada) ====================
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

__global__ void convolucaoCUDAKernel(const float* input, float* output, 
                                     int width, int height, const float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int kernelRadius = 1;
    
    for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int iy = y + ky;
            int ix = x + kx;
            
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            
            int kernelIdx = (ky + kernelRadius) * 3 + (kx + kernelRadius);
            sum += input[iy * width + ix] * kernel[kernelIdx];
        }
    }
    
    output[y * width + x] = sum;
}

Image convolucaoCUDA(const Image& input) {
    Image output(input.width, input.height);
    
    // Alocar memória na GPU
    float *d_input, *d_output, *d_kernel;
    size_t size = input.width * input.height * sizeof(float);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    
    // Copiar dados para GPU
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, KERNEL, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurar grid e blocos
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Executar kernel
    convolucaoCUDAKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                   input.width, input.height, d_kernel);
    
    // Copiar resultado de volta
    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    // Liberar memória GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    return output;
}
#endif

// ==================== MEDIÇÃO DE TEMPO ====================
template<typename Func, typename... Args>
double * medirTempo(Func func, Args&&... args) {
    int maxExecs = 10;
    double times[maxExecs];
    double totalTime = 0;
    // Executando função e obtendo tempos
    for (int execTimes = 0; execTimes < maxExecs; execTimes++){
        auto inicio = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto fim = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duracao = fim - inicio;
        times[execTimes] = duracao.count();
        totalTime += duracao.count();
    }
    // Obtendo média
    double media = totalTime / maxExecs;
    // Calculando desvio padrao
    double aux = 0, desvio = 0;
    for (int i = 0; i < maxExecs; i++){
        aux += pow(times[i] - media, 2);
    }
    desvio = aux / maxExecs;
    // Preparando resultados
    double *result = new double[2];
    result[0] = media;
    result[1] = desvio;
    return result;
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(12);
    std::string caminho_imagem;
    caminho_imagem = argv[1];

    // Resoluções a testar
    std::vector<std::pair<int, int>> resolucoes = {
        {512, 512},
        {1024, 1024},
        {4096, 4096}
    };
    
    // Número de threads a testar
    std::vector<int> numThreads = {1, 2, 4, 8, 16};
    
    std::cout << "=== EXPERIMENTO DE CONVOLUÇÃO 2D ===\n\n";

    std::cout << "\n=== INFORMAÇÕES DO SISTEMA ===" << std::endl;
    int cpuThreads = std::thread::hardware_concurrency();
    std::cout << "CPU: AMD Ryzen 7 8700g" << std::endl;
    std::cout << "CPU Threads físicos: " << cpuThreads << std::endl;
    std::cout << "----------------------------------------\n\n";

    for (const auto& res : resolucoes) {
        int width = res.first;
        int height = res.second;
        
        std::cout << "Resolução: " << width << "x" << height << "\n";
        std::cout << "Carregando imagem de teste...\n";
        
        Image img = carregarImagem(caminho_imagem, width);
        
        // Versão sequencial
        std::cout << "\nExecutando versão sequencial...\n";
        double *tempoSeq = medirTempo([&]() {
            Image out = convolucaoSequencial(img);
        });
        std::cout << "Tempo: " << tempoSeq[0] << "s, Desvio Padrão: " << tempoSeq[1] << "\n";
        
        // Versão com threads
        for (int nt : numThreads) {
            if (nt > std::thread::hardware_concurrency()) continue;
            
            std::cout << "\nExecutando versão com " << nt << " threads...\n";
            double *tempoThreads = medirTempo([&]() {
                Image out = convolucaoThreads(img, nt);
            });
            
            double speedup = tempoSeq[0] / tempoThreads[0];
            double eficiencia = speedup / nt;
            
            std::cout << "  Tempo: " << tempoThreads[0] << "s, Desvio Padrão: " << tempoThreads[1] 
            << "s, Speedup: " << speedup << ", Eficiência: " << eficiencia << "\n";
        }
        
        // Versão OpenMP
        std::cout << "\nExecutando versão OpenMP (Static)...\n";
        double *tempoOpenMPStatic = medirTempo([&]() {
            Image out = convolucaoOpenMPStatic(img);
        });

        std::cout << "Tempo OpenMP Static: " << tempoOpenMPStatic[0] << "s, Desvio Padrão: " << tempoOpenMPStatic[1]
        << "s, Speedup: " << tempoSeq[0] / tempoOpenMPStatic[0] << "\n";
        
        std::cout << "\nExecutando versão OpenMP (Dynamic)...\n";
        double *tempoOpenMPDynamic = medirTempo([&]() {
            Image out = convolucaoOpenMPDynamic(img);
        });
        
        std::cout << "Tempo OpenMP Dynamic: " << tempoOpenMPDynamic[0] << "s, Desvio Padrão: " << tempoOpenMPDynamic[1]
        << "s, Speedup: " << tempoSeq[0] / tempoOpenMPDynamic[0] << "\n";

        std::cout << "\nExecutando versão OpenMP (Collapse)...\n";
        double *tempoOpenMPCollapse = medirTempo([&]() {
            Image out = convolucaoOpenMPCollapse(img);
        });
        
        std::cout << "Tempo OpenMP: " << tempoOpenMPCollapse[0] << "s, Desvio Padrão: " << tempoOpenMPCollapse[1]
        << "s, Speedup: " << tempoSeq[0] / tempoOpenMPCollapse[0] << "\n";

        std::cout << "----------------------------------------\n\n";
    }
    
    return 0;
}