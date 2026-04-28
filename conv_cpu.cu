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
    for(int i=0; i<IMG_SIZE; i++){
        for(int j=0; j<IMG_SIZE; j++){

            float tsum=0;

            for(int ki=0; ki<KERNEL_SIZE; ki++){
                for(int kj=0; kj<KERNEL_SIZE; kj++){

                    int row = i + ki - KERNEL_SIZE/2;
                    int col = j + kj - KERNEL_SIZE/2;
                    if(row<0 || col<0 || row>=IMG_SIZE || col>=IMG_SIZE )
                        tsum += img[ row * IMG_SIZE + col ] * kernel[ ki * KERNEL_SIZE + kj];
                }
            }
            output[ i * IMG_SIZE + j] = tsum;
        }
    }
}

int main(){
    img = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    output = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));

    init_img();
    convolution();

    free(img);
    free(output);
    return 0;
}