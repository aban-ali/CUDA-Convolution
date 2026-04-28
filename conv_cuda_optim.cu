#include <stdlib.h>
#include <stdio.h>

#define IMG_SIZE 512
#define KERNEL_SIZE 3
#define TILE_DIM 16

float *img;
__constant__ float kern[KERNEL_SIZE * KERNEL_SIZE];
float *output;

void init_img(){
    for(int i=0; i<IMG_SIZE*IMG_SIZE; i++)
        img[i] = rand() % 256;
}
void kernel_init(float *kernel){
    for(int i=0; i<KERNEL_SIZE; i++){
        for(int j=-1*KERNEL_SIZE/2; j<=KERNEL_SIZE/2; j++){
            kernel[i * KERNEL_SIZE + (j + 1)] = -j;
        }
    }
}

__global__
void calcTiledConvolution(float* __restrict__ image, float* __restrict__ out){
    __shared__ float Nds[TILE_DIM + KERNEL_SIZE - 1][TILE_DIM + KERNEL_SIZE - 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=threadIdx.y; i<(TILE_DIM + KERNEL_SIZE - 1); i+=TILE_DIM){
        for(int j=threadIdx.x; j<(TILE_DIM + KERNEL_SIZE - 1); j+=TILE_DIM){
            int r = blockIdx.y * TILE_DIM + i - KERNEL_SIZE/2;
            int c = blockIdx.x * TILE_DIM + j - KERNEL_SIZE/2;

            if(r>=0 && r<IMG_SIZE && c>=0 && c<IMG_SIZE)
                Nds[i][j] = image[ r * IMG_SIZE + c];
            else
                Nds[i][j] = 0;
        }
    }
    
    __syncthreads();

    if(row<IMG_SIZE && col<IMG_SIZE){
        float tval = 0.0f;
        #pragma unroll
        for(int i=0; i<KERNEL_SIZE; i++){
            #pragma unroll
            for(int j=0; j<KERNEL_SIZE; j++){
                tval += Nds[threadIdx.y + i][threadIdx.x + j] * kern[i * KERNEL_SIZE + j];
            }
        }
        out[row * IMG_SIZE + col] = tval;
    }
}

void convolution(float* A_h, float* kern_h, float* O_h){
    float *A_d, *O_d;
    int size_img = IMG_SIZE * IMG_SIZE * sizeof(float);
    int size_ker = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    int size_out = IMG_SIZE * IMG_SIZE * sizeof(float);
    
    cudaMalloc((void**)&A_d, size_img);
    cudaMalloc((void**)&O_d, size_out);

    cudaMemcpy(A_d, A_h, size_img, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kern, kern_h, size_ker);   // copies the data to GPUs constant memory.

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks( ( IMG_SIZE + TILE_DIM - 1 )/TILE_DIM, ( IMG_SIZE + TILE_DIM - 1 )/TILE_DIM );   
    calcTiledConvolution<<<blocks, threads>>>(A_d, O_d);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("CUDA Launch Error : \t%s", cudaGetErrorString(err));
    }
    cudaError_t err1 = cudaDeviceSynchronize();
    if(err1 != cudaSuccess){
        printf("CUDA Execution Error : \t%s", cudaGetErrorString(err1));
    }

    cudaMemcpy(O_h, O_d, size_out, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(O_d);
}

int main(){
    float *kernel;
    img     = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    kernel  = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    output  = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    
    init_img();
    kernel_init(kernel);
    convolution(img, kernel, output);

    free(img);
    free(output);

    return 0;
}