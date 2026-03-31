//
// Created by biprarshi on 31/03/2026.
//

#ifndef MATMUL_H
#define MATMUL_H

void matmul(const float *A, const float *B, float *C, const int M, const int K, const int N, const int trans_a, const int trans_b);
void matmul_bias( const float* A, const float* B, const float* bias, float* C, int M, int K, int N);
void add_bias(const float* bias, float* output, int M, int K);
void add_bias_row(const float* bias, float* output, int M, int K);

#endif //MATMUL_H