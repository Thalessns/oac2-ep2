#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <omp.h>

// ==================== CONSTANTES ====================
const int TARGET_SIZES[] = {512, 1024, 4096};
const int NUM_SIZES = 3;

// Kernel CUDA constante
__constant__ float DEVICE_KERNEL[9];

// ==================== ESTRUTURA PARA RESULTADOS ====================
struct Resultado {
    std::string nome;
    double tempo;
    double speedup;
    double eficiencia;
    int numThreads;
};

// ==================== ESTRUTURA IMAGEM ====================
struct Image {
    int width;
    int height;
    std::vector<float> data;
    
    Image(int w = 0, int h = 0) : width(w), height(h) {
        if (w > 0 && h > 0) {
            data.resize(w * h, 0.0f);
        }
    }
    
    // Acesso seguro aos pixels
    float get(int y, int x) const {
        if (y >= 0 && y < height && x >= 0 && x < width) {
            return data[y * width + x];
        }
        return 0.0f;
    }
    
    void set(int y, int x, float value) {
        if (y >= 0 && y < height && x >= 0 && x < width) {
            data[y * width + x] = value;
        }
    }
    
    int totalPixels() const { return width * height; }
    size_t memorySize() const { return data.size() * sizeof(float); }
    
    // Redimensionar imagem usando interpolação bilinear
    Image resize(int newWidth, int newHeight) const {
        Image result(newWidth, newHeight);
        
        float scaleX = static_cast<float>(width) / newWidth;
        float scaleY = static_cast<float>(height) / newHeight;
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;
                
                int x1 = static_cast<int>(srcX);
                int y1 = static_cast<int>(srcY);
                int x2 = std::min(x1 + 1, width - 1);
                int y2 = std::min(y1 + 1, height - 1);
                
                float dx = srcX - x1;
                float dy = srcY - y1;
                
                // Interpolação bilinear
                float val = 
                    get(y1, x1) * (1 - dx) * (1 - dy) +
                    get(y1, x2) * dx * (1 - dy) +
                    get(y2, x1) * (1 - dx) * dy +
                    get(y2, x2) * dx * dy;
                
                result.set(y, x, val);
            }
        }
        
        return result;
    }
};

// ==================== FUNÇÕES PARA IMAGEM PGM ====================
Image loadPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Erro ao abrir: " << filename << std::endl;
        return Image(0, 0);
    }
    
    // Ler cabeçalho
    std::string magic;
    int width, height, maxval;
    
    file >> magic;
    if (magic != "P5" && magic != "P2") {
        std::cerr << "Formato não suportado: " << magic << " (use P5 ou P2)" << std::endl;
        return Image(0, 0);
    }
    
    file >> width >> height >> maxval;
    file.ignore(1); // Pular nova linha
    
    Image img(width, height);
    
    if (magic == "P5") { // Binário
        std::vector<unsigned char> buffer(width * height);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float val = buffer[y * width + x] / 255.0f;
                img.set(y, x, val);
            }
        }
    } else if (magic == "P2") { // ASCII
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel;
                file >> pixel;
                img.set(y, x, pixel / 255.0f);
            }
        }
    }
    return img;
}

// ==================== CONVERSOR DE IMAGENS COMUNS PARA PGM ====================
bool convertToPGM(const std::string& inputFile, const std::string& outputFile) {
    // Verificar extensão
    std::string ext = inputFile.substr(inputFile.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Se já for PGM, copiar
    if (ext == "pgm" || ext == "pbm" || ext == "ppm") {
        std::ifstream src(inputFile, std::ios::binary);
        std::ofstream dst(outputFile, std::ios::binary);
        dst << src.rdbuf();
        return true;
    }
    
    // Para outros formatos, usar ImageMagick (se disponível)
    std::string command = "convert \"" + inputFile + "\" -colorspace Gray \"" + outputFile + "\" 2>/dev/null";
    int result = system(command.c_str());
    
    if (result != 0) {
        std::cerr << "AVISO: Instale ImageMagick para converter automaticamente:" << std::endl;
        std::cerr << "  Ubuntu: sudo apt-get install imagemagick" << std::endl;
        std::cerr << "  Ou converta manualmente: convert imagem.jpg imagem.pgm" << std::endl;
        return false;
    }
    
    return true;
}

// ==================== CARREGAR QUALQUER IMAGEM ====================
Image loadImage(const std::string& filename, int targetSize = 0) {
    // Primeiro verificar se é PGM
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    Image img;
    
    if (ext == "pgm" || ext == "pbm" || ext == "ppm") {
        img = loadPGM(filename);
    } else {
        // Converter para PGM temporário
        std::string tempFile = "temp_conversion.pgm";
        if (convertToPGM(filename, tempFile)) {
            img = loadPGM(tempFile);
            remove(tempFile.c_str()); // Limpar arquivo temporário
        } else {
            std::cerr << "Criando imagem de teste como fallback..." << std::endl;
            // Fallback: criar imagem de teste
            int size = targetSize > 0 ? targetSize : 512;
            img = Image(size, size);
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    float val = ((x + y) % 256) / 255.0f;
                    img.set(y, x, val);
                }
            }
        }
    }
    
    // Redimensionar para tamanho alvo se necessário
    if (targetSize > 0 && (img.width != targetSize || img.height != targetSize)) {
        img = img.resize(targetSize, targetSize);
    }
    
    return img;
}

// ==================== KERNEL CUDA ====================

__global__ void convolutionKernelNoShared(
        const float* input, float* output, 
    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    for (int ky = -1; ky <= 1; ky++) {
        int iy = y + ky;
        // Clamp das bordas na direção y
        if (iy < 0) iy = 0;
        else if (iy >= height) iy = height - 1;
        
        for (int kx = -1; kx <= 1; kx++) {
            int ix = x + kx;
            // Clamp das bordas na direção x
            if (ix < 0) ix = 0;
            else if (ix >= width) ix = width - 1;
            
            int idx = (ky + 1) * 3 + (kx + 1);
            sum += input[iy * width + ix] * DEVICE_KERNEL[idx];
        }
    }
    
    output[y * width + x] = sum;
}


// ==================== KERNEL CUDA COM SHARED MEMORY ====================
__global__ void convolutionKernelShared(
    const float* input, float* output, 
    int width, int height) {
    
    // Coordenadas globais
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Coordenadas locais no bloco
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Dimensões do bloco
    int blockHeight = blockDim.y;
    
    // Dimensões da shared memory com halo
    int sharedWidth = blockDim.x + 2;  // +2 para halo
    int sharedHeight = blockDim.y + 2; // +2 para halo
    
    // Shared memory dinâmica
    extern __shared__ float sharedMem[];
    
    // Calcular índices com halo
    int haloX = blockIdx.x * blockDim.x - 1;
    int haloY = blockIdx.y * blockDim.y - 1;
    
    // Cada thread carrega até 4 elementos (para blocos 16x16)
    for (int i = ly; i < sharedHeight; i += blockHeight) {
        for (int j = lx; j < sharedWidth; j += blockDim.x) {
            int globalY = haloY + i;
            int globalX = haloX + j;
            
            // Clamp das bordas
            if (globalY < 0) globalY = 0;
            else if (globalY >= height) globalY = height - 1;
            
            if (globalX < 0) globalX = 0;
            else if (globalX >= width) globalX = width - 1;
            
            sharedMem[i * sharedWidth + j] = input[globalY * width + globalX];
        }
    }
    
    __syncthreads();
    
    // Se a thread está fora da imagem, retorna
    if (gx >= width || gy >= height) return;
    
    // Convolução usando shared memory
    float sum = 0.0f;
    
    // Índice na shared memory (sem halo para o pixel central)
    int sharedIdx = (ly + 1) * sharedWidth + (lx + 1);
    
    // Usando deslocamentos predefinidos na shared memory
    sum += sharedMem[sharedIdx - sharedWidth - 1] * DEVICE_KERNEL[0];
    sum += sharedMem[sharedIdx - sharedWidth] * DEVICE_KERNEL[1];
    sum += sharedMem[sharedIdx - sharedWidth + 1] * DEVICE_KERNEL[2];
    sum += sharedMem[sharedIdx - 1] * DEVICE_KERNEL[3];
    sum += sharedMem[sharedIdx] * DEVICE_KERNEL[4];
    sum += sharedMem[sharedIdx + 1] * DEVICE_KERNEL[5];
    sum += sharedMem[sharedIdx + sharedWidth - 1] * DEVICE_KERNEL[6];
    sum += sharedMem[sharedIdx + sharedWidth] * DEVICE_KERNEL[7];
    sum += sharedMem[sharedIdx + sharedWidth + 1] * DEVICE_KERNEL[8];
    
    output[gy * width + gx] = sum;
}

// ==================== VERSÃO CUDA SEM SHARED MEMORY ====================
double convolutionCUDA_NoShared(const Image& input, Image& output) {
    // Configurar kernel constante
    float hostKernel[9];
    for (int i = 0; i < 9; i++) hostKernel[i] = 1.0f / 9.0f;  // Média 3x3
    
    cudaMemcpyToSymbol(DEVICE_KERNEL, hostKernel, 9 * sizeof(float));
    
    // Alocar memória GPU
    size_t size = input.memorySize();
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copiar dados de entrada
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);
    
    // Configurar grid/blocos
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, 
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Medir tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolutionKernelNoShared<<<gridSize, blockSize>>>(
        d_input, d_output, input.width, input.height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copiar resultado de volta
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.totalPixels());
    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    // Tempo
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Liberar memória
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / 1000.0f;
}

// ==================== VERSÃO CUDA COM SHARED MEMORY ====================
double convolutionCUDA_Shared(const Image& input, Image& output) {
    // Configurar kernel constante
    float hostKernel[9];
    for (int i = 0; i < 9; i++) hostKernel[i] = 1.0f / 9.0f;  // Média 3x3
    
    cudaMemcpyToSymbol(DEVICE_KERNEL, hostKernel, 9 * sizeof(float));
    
    // Alocar memória GPU
    size_t size = input.memorySize();
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copiar dados de entrada
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);
    
    // Configurar grid/blocos
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, 
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Calcular tamanho da shared memory por bloco (incluindo halo)
    size_t sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);
    
    // Medir tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolutionKernelShared<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, 
                                                                    input.width, input.height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copiar resultado de volta
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.totalPixels());
    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    // Tempo
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Liberar memória
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / 1000.0f;
}

// ==================== FUNÇÃO PARA TESTAR UMA RESOLUÇÃO ====================
void testarResolucao(const Image& original, int targetSize, const std::string& testType, 
                     std::vector<Resultado>& resultadosTotais) {
    std::vector<Resultado> resultados;
    
    std::cout << "=== TESTANDO RESOLUÇÃO " << targetSize << "x" << targetSize << " ===" << std::endl;
    
    // Testar GPU CUDA
    if (testType == "all" || testType == "gpu" || testType == "cuda") {
        // Teste No Shared
        int maxExecs = 10;
        double timesNoShared[maxExecs];
        double totalTimeNoShared = 0;
        std::cout << "\nGPU CUDA (No Shared Memory) Executando..." << std::endl;
        for (int execNumber = 0; execNumber < maxExecs; execNumber++){
            Image gpuResult;
            double timeGPU = convolutionCUDA_NoShared(original, gpuResult);
            timesNoShared[execNumber] = timeGPU;
            totalTimeNoShared += timeGPU;
        }
        double mediaNoShared = totalTimeNoShared / maxExecs;
        double aux1 = 0, desvioNoShared = 0;      
        for (int i = 0; i < maxExecs; i++){
            aux1 += pow(timesNoShared[i] - mediaNoShared, 2);
        }
        desvioNoShared = aux1 / maxExecs;

        std::cout << "Tempo: " << mediaNoShared << "s, Desvio Padrão: " << desvioNoShared << "s, " 
                  << std::endl;

        // Teste Shared
        double timesShared[maxExecs];
        double totalTimeShared = 0;
        std::cout << "\nGPU CUDA (Shared Memory) Executando..." << std::endl;
        for (int execNumber = 0; execNumber < maxExecs; execNumber++){
            Image gpuResult;
            double timeGPU = convolutionCUDA_Shared(original, gpuResult);
            timesShared[execNumber] = timeGPU;
            totalTimeShared += timeGPU;
        }
        double mediaShared = totalTimeShared / maxExecs;
        double aux2 = 0, desvioShared = 0;      
        for (int i = 0; i < maxExecs; i++){
            aux2 += pow(timesShared[i] - mediaShared, 2);
        }
        desvioShared = aux2 / maxExecs;

        std::cout << "Tempo: " << mediaShared << "s, Desvio Padrão: " << desvioShared << "s, " 
                  << std::endl;
    }
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== CONVOLUÇÃO 2D - GPU ===" << std::endl;
    
    // Verificar argumentos
    if (argc < 2) {
        std::cout << "\nUso: " << argv[0] << " <imagem> [tipo_teste]" << std::endl;
        std::cout << "Exemplos:" << std::endl;
        std::cout << "  " << argv[0] << " imagem.pgm" << std::endl;
        std::cout << "  " << argv[0] << " imagem.jpg all" << std::endl;
        std::cout << "  " << argv[0] << " imagem.png threads" << std::endl;
        return 1;
    }
    
    std::string imagePath = argv[1];
    std::string testType = (argc > 2) ? argv[2] : "all";
    
    // Informações do sistema
    std::cout << "\n=== INFORMAÇÕES DO SISTEMA ===" << std::endl;
    int cpuThreads = std::thread::hardware_concurrency();
    std::cout << "CPU: AMD Ryzen 7 8700g" << std::endl;
    std::cout << "CPU Threads físicos: " << cpuThreads << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memória GPU: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Máx. Threads por Bloco: " << prop.maxThreadsPerBlock << std::endl;
    
    // Armazenar todos os resultados
    std::vector<Resultado> resultadosTotais;
    
    // Testar todas as resoluções
    for (int sizeIndex = 0; sizeIndex < NUM_SIZES; sizeIndex++) {
        int targetSize = TARGET_SIZES[sizeIndex];
        
        // Carregar e redimensionar imagem
        std::cout << "\n\n" << std::string(60, '=') << std::endl;
        
        Image original = loadImage(imagePath, targetSize);
        
        if (original.width == 0 || original.height == 0) {
            std::cerr << "Erro ao carregar imagem!" << std::endl;
            continue;
        }
        
        testarResolucao(original, targetSize, testType, resultadosTotais);
    }

    cudaDeviceReset();
    return 0;
}
