#include <math.h>
#include <string.h>
#include "matmul.h"

// Extract image patches into column matrix.
// input - [c , h , w] -> output [h_out * w_out , C * kh * kw]
void im2col(const float *input , float *col, int C, int H, int W, int KH, int KW, int stride, int pad, int H_out, int W_out) {
    for (int i = 0 ; i < H_out ; i++) {
        for (int j = 0 ; j < W_out ; j++) {
            int oh = i * W_out + j;
            int ow = 0;

            for (int c = 0 ; c < C ; c++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int input_h = i * stride - pad + kh;
                        int input_w = j * stride - pad + kw;
                        int input_idx = c * (H * W) + input_h * W + input_w;

                        if (input_h < 0 || input_h >= H || input_w < 0 || input_w >= W) { // if selected submatrix out of bounds
                            col[oh * (C * KH * KW) +  ow] = 0.0;
                        } else {
                            col[oh * (C * KH * KW) + ow] = input[input_idx];
                        }
                        ow++;
                    }
                }
            }
        }
    }
}

void conv2d_forward(const float *input , const float *weights , const float *bias , float *output , float *col_buf , int C, int H, int W , int F, int KH, int KW, int stride, int pad) {
    int H_out = floor((H + 2*pad - KH) / stride) + 1;
    int W_out = floor((W + 2*pad - KW) / stride) + 1;

    im2col(input , col_buf , C , H , W , KH , KW , stride , pad , H_out , W_out);
    matmul(weights , col_buf , output , F , C * KH * KW , H_out * W_out , 0 , 1);  // W @ col^T (to reshape final output to [F, H_outW_out])
    add_bias_row(bias , output , F , H_out*W_out); // add bias[f] to each element in filter f's row
}

// for N images
void conv2d_forward_batch(const float *input , const float *weights , const float *bias , float *output , float *col_buf , int N , int C, int H, int W , int F, int KH, int KW, int stride, int pad) {
    int H_out = floor((H + 2*pad - KH) / stride) + 1;
    int W_out = floor((W + 2*pad - KW) / stride) + 1;
    for (int i = 0 ; i < N ; i++) {
        conv2d_forward(input + i * (C * H * W) , weights , bias , output + i * (F * H_out * W_out) , col_buf , C , H , W , F , KH , KW , stride , pad);
    }
}


