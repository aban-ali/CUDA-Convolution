#include <stdlib.h>
#include <stdio.h>

#define IMG_SIZE 512
#define FILTER_SIZE 3

float *img;
float *filter;
float *output;

void init_img(){
    for(int i=0; i<IMG_SIZE*IMG_SIZE; i++)
        img[i] = rand() % 256;
}
void init_filter(){
    for(int i=0; i<FILTER_SIZE*FILTER_SIZE; i++){
        filter[i] = rand() % FILTER_SIZE;
        if(((int)filter[i]) & 1)
            filter[i] *= -1;
    }
}

__global__
void calcConvolution(float* image, float* conv_filter, float* out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<IMG_SIZE && col<IMG_SIZE){
        
        float val = 0;

        for(int i=0; i<FILTER_SIZE; i++){
            for(int j=0; j<FILTER_SIZE; j++){

                int krow = row + i - FILTER_SIZE/2;
                int kcol = col + j - FILTER_SIZE/2;

                if(krow>=0 && krow<IMG_SIZE && kcol>=0 && kcol<IMG_SIZE)
                    val += image[ krow * IMG_SIZE + kcol ] * conv_filter[ i * FILTER_SIZE + j ];
            }
        }

        out[row * IMG_SIZE + col] = val;
    }
}

void convolution(float* A_h, float* O_h){
    float        *A_d, *filter_d, *O_d;
    int          size_img = IMG_SIZE * IMG_SIZE * sizeof(float);
    int          size_filter = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    int          size_out = IMG_SIZE * IMG_SIZE * sizeof(float);
    cudaEvent_t  start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&A_d, size_img);
    cudaMalloc((void**)&filter_d, size_filter);
    cudaMalloc((void**)&O_d, size_out);

    cudaMemcpy(A_d, A_h, size_img, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter, size_filter, cudaMemcpyHostToDevice);

    dim3 threads(8, 8);
    dim3 blocks( (IMG_SIZE + threads.x - 1)/threads.x, (IMG_SIZE + threads.y - 1)/threads.y);   
    calcConvolution<<<blocks, threads>>>(A_d, filter_d, O_d);
    cudaDeviceSynchronize();

    float total = 0.0f;
    for(int t=0; t<10; t++){
        cudaEventRecord(start);
        calcConvolution<<<blocks, threads>>>(A_d, filter_d, O_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    printf("Average time of 10 runs: %f ms\n", total/10);
    
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
    cudaFree(filter_d);
    cudaFree(O_d);
}

int main(){
    img = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    output = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    
    init_img();
    init_filter();
    convolution(img, output);

    free(img);
    free(output);

    return 0;
}