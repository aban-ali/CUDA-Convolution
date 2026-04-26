#include <stdlib.h>

#define IMG_SIZE 512
#define KERNEL_SIZE 3

float *img;
float kernel[KERNEL_SIZE * KERNEL_SIZE] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
float *output;

void init_img(){
    for(int i=0; i<IMG_SIZE*IMG_SIZE; i++)
        img[i] = rand() % 256;
}

__global__
void calcConvolution(float* image, float* kern, float* out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    if(row<IMG_SIZE-KERNEL_SIZE+1 && col<IMG_SIZE-KERNEL_SIZE+1){
        for(int i=0; i<KERNEL_SIZE; i++){
            for(int j=0; j<KERNEL_SIZE; j++){
                val += image[ (row+i) * IMG_SIZE + (col+j)] * kern[ i * KERNEL_SIZE + j ];
            }
        }
        out[row * (IMG_SIZE - KERNEL_SIZE + 1) + col] = val;
    }
}

void convolution(float* A_h, float* O_h){
    float *A_d, *ker_d, *O_d;
    int size_img = IMG_SIZE * IMG_SIZE * sizeof(float);
    int size_ker = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    int size_out = (IMG_SIZE - KERNEL_SIZE + 1) * (IMG_SIZE - KERNEL_SIZE + 1) * sizeof(float);
    
    cudaMalloc((void**)&A_d, size_img);
    cudaMalloc((void**)&ker_d, size_ker);
    cudaMalloc((void**)&O_d, size_out);

    cudaMemcpy(A_d, A_h, size_img, cudaMemcpyHostToDevice);
    cudaMemcpy(ker_d, kernel, size_ker, cudaMemcpyHostToDevice);

    dim3 threads(8, 8);
    dim3 blocks(IMG_SIZE/8, IMG_SIZE/8);   
    calcConvolution<<<blocks, threads>>>(A_d, ker_d, O_d);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("CUDA Error : \t%s", cudaGetErrorString(err));
    }

    cudaMemcpy(O_h, O_d, size_out, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(ker_d);
    cudaFree(O_d);
}

int main(){
    img = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    output = (float*)malloc( (IMG_SIZE - KERNEL_SIZE + 1) * 
                            (IMG_SIZE - KERNEL_SIZE + 1) * sizeof(float));
    
    init_img();
    convolution(img, output);

    free(img);
    free(output);

    return 0;
}