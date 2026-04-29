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
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    __shared__ float Nds[TILE_DIM][TILE_DIM];

    if(row<IMG_SIZE && col<IMG_SIZE)
        Nds[threadIdx.y][threadIdx.x] = image[ row * IMG_SIZE + col ];
    else
        Nds[threadIdx.y][threadIdx.x] = 0;
    
    __syncthreads();
    
    if(row<IMG_SIZE && col<IMG_SIZE){
            float val = 0;
        for(int i=-FILTER_SIZE/2; i<=FILTER_SIZE/2; i++){
            for(int j=-FILTER_SIZE/2; j<=FILTER_SIZE/2; j++){
                int krow = threadIdx.y + i;
                int kcol = threadIdx.x + j;

                if(krow>=0 && krow<TILE_DIM && kcol>=0 && kcol<TILE_DIM)
                    val += Nds[krow][kcol] * conv_filter[ (i+FILTER_SIZE/2) * FILTER_SIZE + (j+FILTER_SIZE/2) ];
                else if(row+i>=0 && row+i<IMG_SIZE && col+j>=0 && col+j<IMG_SIZE)
                    val += conv_filter[(i+FILTER_SIZE/2) * FILTER_SIZE + (j+FILTER_SIZE/2)] * 
                            image[ ( row + i )*IMG_SIZE + ( col + j ) ];  // will be cached in, perhaps, L2 memory
            }
        }
        out[row * IMG_SIZE + col] = val;
    }
}

void convolution(float* A_h, float* kern_h, float* O_h){
    float        *A_d, *O_d;
    int          size_img = IMG_SIZE * IMG_SIZE * sizeof(float);
    int          size_ker = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    int          size_out = IMG_SIZE * IMG_SIZE * sizeof(float);
    cudaEvent_t  start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void**)&A_d, size_img);
    cudaMalloc((void**)&O_d, size_out);

    cudaMemcpy(A_d, A_h, size_img, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(conv_filter, kern_h, size_ker);   // copies the data to GPUs constant memory.

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks( ( IMG_SIZE + TILE_DIM - 1 )/TILE_DIM, ( IMG_SIZE + TILE_DIM - 1 )/TILE_DIM );   
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