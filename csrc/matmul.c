#include <stdio.h>
#include <stdlib.h>

#define INDEX(i , j , B) ((i) * (B) + (j))

// C = A @ B
void matmul(const float *A, const float *B, float *C, const int M, const int K, const int N, const int trans_a, const int trans_b) {
    float accumulator = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            accumulator = 0;

            for (int k = 0; k < K; k++) {
                int a_idx , b_idx;
                if (trans_a == 1) {
                    a_idx = INDEX(k, i, M);
                } else {
                    a_idx = INDEX(i, k, K);
                }

                if (trans_b == 1) {
                    b_idx = INDEX(j, k, K);
                } else {
                    b_idx = INDEX(k, j, N);
                }

                accumulator += A[a_idx] * B[b_idx];
            }

            int c_idx = INDEX(i, j, N);
            C[c_idx] = accumulator;
        }
    }
}


// C = A @ B + bias
void matmul_bias( const float* A, const float* B, const float* bias, float* C, int M, int K, int N) {
    matmul(A , B , C , M , K , N , 0 , 0);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int c_idx = INDEX(i, j, N);
            C[c_idx] += bias[j];
        }
    }
}

void add_bias(const float* bias, float* output, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            int c_idx = INDEX(i, j, K);
            output[c_idx] += bias[j];
        }
    }
}

void add_bias_row(const float* bias, float* output, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            int c_idx = INDEX(i, j, K);
            output[c_idx] += bias[i];
        }
    }
}