// #include <stdio.h>
#include <stdlib.h>

#define IMG_SIZE 512
#define KERNEL_SIZE 3

float *img;
float kernel[ KERNEL_SIZE * KERNEL_SIZE ] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
float *output;

void init_img(){
    for(int i=0; i< IMG_SIZE * IMG_SIZE; i++){
            img[i] = rand() % 256;
    }
}

void convolution(){
    int out_size = IMG_SIZE - KERNEL_SIZE + 1;
    for(int i=0; i<out_size; i++){
        for(int j=0; j<out_size; j++){

            float tsum=0;

            for(int ki=0; ki<KERNEL_SIZE; ki++){
                for(int kj=0; kj<KERNEL_SIZE; kj++){
                    tsum += img[ (i+ki) * IMG_SIZE + (j+kj) ] * 
                            kernel[ ki * KERNEL_SIZE + kj];
                }
            }
            output[ i * out_size + j] = tsum;
        }
    }
}

int main(){
    img = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    output = (float*)malloc((IMG_SIZE - KERNEL_SIZE + 1) * 
                            (IMG_SIZE - KERNEL_SIZE + 1) * sizeof(float));
    init_img();
    convolution();

    free(img);
    free(output);
    return 0;
}