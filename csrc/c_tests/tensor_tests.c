#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    int shape[] = {2 , 3 , 4};
    int *shape_ptr = shape;
    Tensor *t = tensor_alloc(3 , shape_ptr);

    int offset[] = {1 , 2 , 3};
    int offset123 = tensor_offset(t , offset);

    t->data[offset123] = 42;
    int offset_manual = 1*12 + 2*4 + 3;

    assert(t->data[offset_manual] == 42);
    printf("tensor test passed");

    tensor_free(t);
}