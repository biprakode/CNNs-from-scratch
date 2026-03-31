#include <math.h>

#include "activations.h"
// Element-wise ReLU save mask
void relu_forward(const float *input , float *output , int *mask, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = RELU(input[i]);
        mask[i] = (output[i] > 0);
    }
}

void leaky_relu_forward(const float *input , float *output , float *mask, int size , float alpha) {
    for (int i = 0; i < size; i++) {
        output[i] = Leaky_RELU(input[i] , alpha);
        mask[i] = input[i] > 0 ? 1.0f : alpha;
    }
}

// maxpool, save highest indices in indices for backward
void maxpool2d_forward(const float *input , float *output , int *indices , int C , int H , int W , int PH , int PW , int stride) {
    int W_in = W * stride + PW - stride;
    int H_in = H * stride + PH - stride;

    for (int c = 0; c < C; c++) {
        for (int oh = 0 ; oh < H ; oh++) {
            for (int ow = 0 ; ow < W ; ow++) {
                float val = -INFINITY;
                int max_idx = 0;
                for (int ph = 0 ; ph < PH ; ph++) {
                    for (int pw = 0 ; pw < PW ; pw++) {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;

                        if (input[c * H_in * W_in + ih * W_in + iw] > val) {
                            val = input[c * H_in * W_in + ih * W_in + iw];
                            max_idx = c * H_in * W_in + ih * W_in + iw;
                        }
                    }
                }
                output[c * H * W + oh * W + ow] = val;
                indices[c * H * W + oh * W + ow] = max_idx;
            }
        }
    }
}

void relu_forward_batch(const float *input , float *output , int *mask, int total_size) {
    relu_forward(input, output, mask, total_size);
}

void leaky_relu_forward_batch(const float *input , float *output , float *mask, int total_size , float alpha) {
    leaky_relu_forward(input, output, mask, total_size, alpha);
}

