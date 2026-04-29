#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMG_SIZE 2048
#define FILTER_SIZE 7

float *img;
float *filter;
float *output;

void init_img(){
    for(int i=0; i< IMG_SIZE * IMG_SIZE; i++){
            img[i] = rand() % 256;
    }
}
void init_filter(){
    for(int i=0; i<FILTER_SIZE*FILTER_SIZE; i++){
        filter[i] = rand() % FILTER_SIZE;
        if(((int)filter[i]) & 1)
            filter[i] *= -1;
    }
}

void convolution(){
    for(int i=0; i<IMG_SIZE; i++){
        for(int j=0; j<IMG_SIZE; j++){

            float tsum=0;

            for(int ki=0; ki<FILTER_SIZE; ki++){
                for(int kj=0; kj<FILTER_SIZE; kj++){

                    int row = i + ki - FILTER_SIZE/2;
                    int col = j + kj - FILTER_SIZE/2;
                    if(row>=0 && col>=0 && row<IMG_SIZE && col<IMG_SIZE )
                        tsum += img[ row * IMG_SIZE + col ] * filter[ ki * FILTER_SIZE + kj];
                }
            }
            output[ i * IMG_SIZE + j] = tsum;
        }
    }
}

int main(){
    img = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    output = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));

    init_img();
    init_filter();
    convolution();

    double total = 0;
    for(int t = 0; t < 10; t++){
        clock_t start = clock();
        convolution();
        clock_t end = clock();

        total += (double)(end - start);
    }
    double time_ms = (total / 10.0) * 1000.0 / CLOCKS_PER_SEC;
    printf("Avg CPU time: %f ms\n", time_ms);

    free(img);
    free(output);
    return 0;
}