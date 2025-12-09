// convolucao_completa_com_metricas.cu - CUDA + C++ com Speedup e Eficiência
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

// ==================== CONSTANTES ====================
const int TARGET_SIZES[] = {512, 1024, 4096};
const int NUM_SIZES = 3;
const int NUM_THREADS_TESTS[] = {1, 2, 4, 8, 16};

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
__global__ void convolutionKernelSimple(const float* input, float* output, 
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int iy = y + ky;
            int ix = x + kx;
            
            // Clamp das bordas
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            
            int idx = (ky + 1) * 3 + (kx + 1);
            sum += input[iy * width + ix] * DEVICE_KERNEL[idx];
        }
    }
    
    output[y * width + x] = sum;
}

// ==================== VERSÃO CPU SEQUENCIAL ====================
Image convolutionCPU(const Image& input) {
    Image output(input.width, input.height);
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int iy = y + ky;
                    int ix = x + kx;
                    
                    if (iy < 0) iy = 0;
                    if (iy >= input.height) iy = input.height - 1;
                    if (ix < 0) ix = 0;
                    if (ix >= input.width) ix = input.width - 1;
                    
                    sum += input.get(iy, ix) * (1.0f / 9.0f);
                }
            }
            output.set(y, x, sum);
        }
    }
    
    return output;
}

// ==================== VERSÃO CPU COM THREADS ====================
Image convolutionCPUThreads(const Image& input, int numThreads) {
    Image output(input.width, input.height);
    std::vector<std::thread> threads;
    
    auto processRows = [&](int startRow, int endRow) {
        for (int y = startRow; y < endRow; y++) {
            for (int x = 0; x < input.width; x++) {
                float sum = 0.0f;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int iy = y + ky;
                        int ix = x + kx;
                        
                        if (iy < 0) iy = 0;
                        if (iy >= input.height) iy = input.height - 1;
                        if (ix < 0) ix = 0;
                        if (ix >= input.width) ix = input.width - 1;
                        
                        sum += input.get(iy, ix) * (1.0f / 9.0f);
                    }
                }
                output.set(y, x, sum);
            }
        }
    };
    
    int rowsPerThread = input.height / numThreads;
    int currentRow = 0;
    
    for (int i = 0; i < numThreads; i++) {
        int endRow = (i == numThreads - 1) ? input.height : currentRow + rowsPerThread;
        threads.emplace_back(processRows, currentRow, endRow);
        currentRow = endRow;
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return output;
}

// ==================== VERSÃO CUDA ====================
float convolutionCUDA(const Image& input, Image& output) {
    // Configurar kernel
    float hostKernel[9];
    for (int i = 0; i < 9; i++) hostKernel[i] = 1.0f / 9.0f;
    
    cudaMemcpyToSymbol(DEVICE_KERNEL, hostKernel, 9 * sizeof(float));
    
    // Alocar memória GPU
    size_t size = input.memorySize();
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copiar dados
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);
    
    // Configurar grid/blocos
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + 15) / 16, (input.height + 15) / 16);
    
    // Medir tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolutionKernelSimple<<<gridSize, blockSize>>>(d_input, d_output, 
                                                    input.width, input.height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copiar resultado
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

// ==================== FUNÇÕES PARA CÁLCULO DE MÉTRICAS ====================
// Calcular Speedup: S = T_seq / T_paralelo
double calcularSpeedup(double tempoSequencial, double tempoParalelo) {
    if (tempoParalelo <= 0) return 0.0;
    return tempoSequencial / tempoParalelo;
}

// Calcular Eficiência: E = S / N_threads
double calcularEficiencia(double speedup, int numThreads) {
    if (numThreads <= 0) return 0.0;
    return speedup / numThreads;
}

// ==================== FUNÇÃO PARA MEDIR TEMPO ====================
template<typename Func, typename... Args>
double measureTime(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ==================== FUNÇÃO PARA EXIBIR RESULTADOS EM TABELA ====================
void exibirTabelaResultados(const std::vector<Resultado>& resultados, double tempoSequencial) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TABELA DE RESULTADOS (Baseline Sequencial: " << std::fixed << std::setprecision(4) 
              << tempoSequencial << "s)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(20) << "VERSÃO" 
              << std::setw(10) << "THREADS" 
              << std::setw(15) << "TEMPO (s)" 
              << std::setw(15) << "SPEEDUP" 
              << std::setw(15) << "EFICIÊNCIA" 
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& res : resultados) {
        std::cout << std::left << std::setw(20) << res.nome
                  << std::setw(10) << res.numThreads
                  << std::fixed << std::setprecision(4)
                  << std::setw(15) << res.tempo
                  << std::setw(15) << res.speedup
                  << std::setw(15) << res.eficiencia
                  << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
}

// ==================== FUNÇÃO PARA TESTAR UMA RESOLUÇÃO ====================
void testarResolucao(const Image& original, int targetSize, const std::string& testType, 
                     std::vector<Resultado>& resultadosTotais) {
    std::vector<Resultado> resultados;
    double tempoSequencial = 0;
    
    std::cout << "\n=== TESTANDO RESOLUÇÃO " << targetSize << "x" << targetSize << " ===" << std::endl;
    std::cout << "Imagem: " << original.width << "x" << original.height 
              << " (" << original.totalPixels() << " pixels)" << std::endl;
    
    // Testar CPU Sequencial (Baseline)
    if (testType == "all" || testType == "cpu" || testType == "seq") {
        std::cout << "\n[CPU Sequencial] Executando..." << std::endl;
        tempoSequencial = measureTime([&]() {
            Image result = convolutionCPU(original);
        });
        
        Resultado res;
        res.nome = "CPU Sequencial";
        res.tempo = tempoSequencial;
        res.speedup = 1.0; // Baseline
        res.eficiencia = 1.0;
        res.numThreads = 1;
        resultados.push_back(res);
        
        std::cout << "Tempo: " << std::fixed << std::setprecision(4) << tempoSequencial << " segundos" << std::endl;
    }
    
    // Testar CPU com Threads
    if (testType == "all" || testType == "cpu" || testType == "threads") {
        for (int numThreads : NUM_THREADS_TESTS) {
            if (numThreads > std::thread::hardware_concurrency()) continue;
            
            std::cout << "\n[CPU Threads: " << numThreads << "] Executando..." << std::endl;
            double timeThreads = measureTime([&]() {
                Image result = convolutionCPUThreads(original, numThreads);
            });
            
            Resultado res;
            res.nome = "CPU Threads";
            res.tempo = timeThreads;
            res.speedup = calcularSpeedup(tempoSequencial, timeThreads);
            res.eficiencia = calcularEficiencia(res.speedup, numThreads);
            res.numThreads = numThreads;
            resultados.push_back(res);
            
            std::cout << "Tempo: " << std::fixed << std::setprecision(4) << timeThreads << " segundos" 
                      << " | Speedup: " << res.speedup 
                      << " | Eficiência: " << res.eficiencia << std::endl;
        }
    }
    
    // Testar GPU CUDA
    if (testType == "all" || testType == "gpu" || testType == "cuda") {
        std::cout << "\n[GPU CUDA] Executando..." << std::endl;
        Image gpuResult;
        float timeGPU = convolutionCUDA(original, gpuResult);
        
        // Estimar threads equivalentes (aproximação para cálculo de eficiência)
        int threadsGPU = 1024; // Threads por bloco típicas * número de blocos
        
        Resultado res;
        res.nome = "GPU CUDA";
        res.tempo = timeGPU;
        res.speedup = calcularSpeedup(tempoSequencial, timeGPU);
        res.eficiencia = res.speedup; // Para GPU, eficiência geralmente = speedup (não normalizada por threads)
        res.numThreads = threadsGPU;
        resultados.push_back(res);
        
        std::cout << "Tempo: " << std::fixed << std::setprecision(4) << timeGPU << " segundos" 
                  << " | Speedup: " << res.speedup << std::endl;
    }
    
    // Exibir tabela para esta resolução
    exibirTabelaResultados(resultados, tempoSequencial);
    
    // Adicionar aos resultados totais (marcando a resolução)
    for (auto& res : resultados) {
        res.nome = res.nome + " (" + std::to_string(targetSize) + ")";
        resultadosTotais.push_back(res);
    }
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== CONVOLUÇÃO 2D - COMPARAÇÃO COM SPEEDUP E EFICIÊNCIA ===" << std::endl;
    std::cout << "Formatos suportados: PGM (recomendado), JPEG, PNG (via conversão)" << std::endl;
    
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
        std::cout << "CARREGANDO IMAGEM PARA " << targetSize << "x" << targetSize << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        Image original = loadImage(imagePath, targetSize);
        
        if (original.width == 0 || original.height == 0) {
            std::cerr << "Erro ao carregar imagem!" << std::endl;
            continue;
        }
        
        testarResolucao(original, targetSize, testType, resultadosTotais);
    }
    
    // Resumo final
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "RESUMO FINAL - TODAS AS RESOLUÇÕES" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::setw(25) << "VERSÃO (RESOLUÇÃO)"
              << std::setw(10) << "THREADS" 
              << std::setw(15) << "TEMPO (s)" 
              << std::setw(15) << "SPEEDUP" 
              << std::setw(15) << "EFICIÊNCIA" 
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& res : resultadosTotais) {
        std::cout << std::left << std::setw(25) << res.nome
                  << std::setw(10) << res.numThreads
                  << std::fixed << std::setprecision(4)
                  << std::setw(15) << res.tempo
                  << std::setw(15) << res.speedup
                  << std::setw(15) << res.eficiencia
                  << std::endl;
    }
    
    // Análise comparativa
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "ANÁLISE COMPARATIVA" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Encontrar melhor speedup por resolução
    for (int sizeIndex = 0; sizeIndex < NUM_SIZES; sizeIndex++) {
        int targetSize = TARGET_SIZES[sizeIndex];
        double maxSpeedup = 0;
        std::string melhorVersao;
        
        for (const auto& res : resultadosTotais) {
            if (res.nome.find(std::to_string(targetSize)) != std::string::npos) {
                if (res.speedup > maxSpeedup) {
                    maxSpeedup = res.speedup;
                    melhorVersao = res.nome;
                }
            }
        }
        
        if (maxSpeedup > 0) {
            std::cout << "Resolução " << targetSize << "x" << targetSize << ":" << std::endl;
            std::cout << "  Melhor Speedup: " << std::fixed << std::setprecision(2) 
                      << maxSpeedup << "x (" << melhorVersao << ")" << std::endl;
            
            // Análise de escalabilidade
            if (melhorVersao.find("Threads") != std::string::npos) {
                std::cout << "  Tipo: Paralelismo CPU com Threads" << std::endl;
            } else if (melhorVersao.find("CUDA") != std::string::npos) {
                std::cout << "  Tipo: Paralelismo GPU (CUDA)" << std::endl;
            } else {
                std::cout << "  Tipo: Sequencial" << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    cudaDeviceReset();
    return 0;
}