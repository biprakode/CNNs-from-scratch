#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

#include <string.h>

Tensor * tensor_alloc(int ndim, int *shape) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    int size = 1;
    for (int i = 0 ; i<ndim ; i++) {
        size *= shape[i];
    }
    t->shape = (int *) malloc(sizeof(int)*ndim);
    memcpy(t->shape , shape, ndim * sizeof(int));
    t->data = (float*) calloc(size , sizeof(float));
    t->size = size;
    t->ndim = ndim;
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t);
}

int tensor_offset(const Tensor *t, int *idx) {
    // [N][C][H][W] maps to offset n*C*H*W + c*H*W + h*W + w
    int offset = 0 , mult;
    for (int i = 0 ; i<t->ndim ; i++) {
        mult = 1;
        for (int j = i+1 ; j < t->ndim ; j++) {
            mult *= t->shape[j];
        }
        offset += mult * idx[i];
    }
    return offset;
}

void tensor_print(const Tensor *t, int max_elems) {
    printf("Data = ");
    for (int i = 0 ; i<max_elems ; i++) {
        printf("%0.4f , " , t->data[i]);
    }
    printf("\nDimensions = ");
    for (int i = 0 ; i<t->ndim ; i++) {
        printf("%d , ", t->shape[i]);
    }
    printf("\nNdim = %d\nSize = %d" , t->ndim, t->size);
    // claude code pretty printify
}
