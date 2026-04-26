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

void conv(float* A_h, float* O_h){

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