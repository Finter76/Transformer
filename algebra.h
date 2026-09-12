#ifndef ALGEBRA_H_
#define ALGEBRA_H_

#include "tokenizer.h"

typedef struct {
    float *data;
    int size;
} Vector;

typedef struct{
    float *data;
    int rows;
    int cols;
} Matrix;



#endif
