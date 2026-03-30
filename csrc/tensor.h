//
// Created by biprarshi on 30/03/2026.
//

#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int *shape;
    int ndim;
    int size;
} Tensor;

// Allocate a tensor with given shape. Zeros the data.
Tensor* tensor_alloc(int ndim, int* shape);

// Free tensor and its internal arrays.
void tensor_free(Tensor* t);

// Compute flat index from multi-dimensional indices.
// idx is an array of length ndim.
int tensor_offset(const Tensor* t, int* idx);

// Print tensor shape and first few elements (for debugging).
void tensor_print(const Tensor* t, int max_elems);

#endif