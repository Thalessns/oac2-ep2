Este projeto tem como objetivo comparar o tempo de execução de um algoritmo de Convolução 2D entre CPU (Sequencial), CPU (OpenMP), CPU (Threads Explicitas) e GPU.

O projeto foi desenvolvido em linux, utilizando uma GPU da nvidia.
Como instalar os requisitos para rodar:
```shell
sudo apt-get update
sudo apt-get install -y \
    g++ \
    make \
    build-essential \
    libomp-dev \
    libpthread-stubs0-dev \
    nvidia-cuda-toolkit  # se tiver GPU NVIDIA
```

Para rodar a aplicação, compile os arquivos da seguinte forma, estando na raiz do projeto:
```shell
make all
```

Para executar, escolha o arquivo desejado `convolucao_cpu` ou `convolucao_cuda` e também o `caminho para uma imagem` que será processada.
```shell
./convolucao_cpu imagens/dog.jpg
```
```shell
./convolucao_cuda imagens/dog.jpg
```
