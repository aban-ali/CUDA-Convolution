#include <stdlib.h>
#include <stdio.h>

#define IMG_SIZE 512
#define FILTER_SIZE 3
#define TILE_DIM 16

float *img;
__constant__ float conv_filter[FILTER_SIZE * FILTER_SIZE];
float *output;

void init_img(){
    for(int i=0; i<IMG_SIZE*IMG_SIZE; i++)
        img[i] = rand() % 256;
}
void init_filter(float *filter){
    for(int i=0; i<FILTER_SIZE*FILTER_SIZE; i++){
        filter[i] = rand() % FILTER_SIZE;
        if(((int)filter[i]) & 1)
            filter[i] *= -1;
    }
}

__global__
void calcTiledConvolution(float* image, float* out){
    int row = blockIdx.y * (TILE_DIM - FILTER_SIZE + 1) + threadIdx.y - FILTER_SIZE/2;
    int col = blockIdx.x * (TILE_DIM - FILTER_SIZE + 1) + threadIdx.x - FILTER_SIZE/2;
    __shared__ float Nds[TILE_DIM][TILE_DIM];

    if(row>=0 && row<IMG_SIZE && col>=0 && col<IMG_SIZE)
        Nds[threadIdx.y][threadIdx.x] = image[ row * IMG_SIZE + col ];
    else
        Nds[threadIdx.y][threadIdx.x] = 0;
    
    __syncthreads();
    
    if(row>=0 && row<IMG_SIZE && col>=0 && col<IMG_SIZE){
        int tile_row = threadIdx.y - FILTER_SIZE/2;
        int tile_col = threadIdx.x - FILTER_SIZE/2;

        if( tile_row>=0 && tile_row<(TILE_DIM - FILTER_SIZE + 1) && 
            tile_col>=0 && tile_col<(TILE_DIM - FILTER_SIZE + 1) ){

            float val = 0;

            for(int i=0; i<FILTER_SIZE; i++){
                for(int j=0; j<FILTER_SIZE; j++){
                    val += Nds[tile_row + i][tile_col + j] * conv_filter[ i * FILTER_SIZE + j ];
                }
            }
            out[(row + FILTER_SIZE/2) * IMG_SIZE + (col + FILTER_SIZE/2)] = val;
        }
    }
}

void convolution(float* A_h, float* kern_h, float* O_h){
    float        *A_d, *O_d;
    int          size_img = IMG_SIZE * IMG_SIZE * sizeof(float);
    int          size_ker = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    int          size_out = IMG_SIZE * IMG_SIZE * sizeof(float);
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void**)&A_d, size_img);
    cudaMalloc((void**)&O_d, size_out);

    cudaMemcpy(A_d, A_h, size_img, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(conv_filter, kern_h, size_ker);   // copies the data to GPUs constant memory.

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks( ( IMG_SIZE + (TILE_DIM - FILTER_SIZE + 1) - 1)/( TILE_DIM - FILTER_SIZE + 1 ), 
                ( IMG_SIZE + (TILE_DIM - FILTER_SIZE + 1) - 1)/( TILE_DIM - FILTER_SIZE + 1 ));   
    calcTiledConvolution<<<blocks, threads>>>(A_d, O_d);
    cudaDeviceSynchronize();
    
    float total = 0.0f;
    for(int t=0; t<10; t++){
        cudaEventRecord(start);   
        calcTiledConvolution<<<blocks, threads>>>(A_d, O_d);
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
    cudaFree(O_d);
}

int main(){
    float *filter;
    img     = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    filter  = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    output  = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    
    init_img();
    init_filter(filter);
    convolution(img, filter, output);

    free(img);
    free(output);

    return 0;
}