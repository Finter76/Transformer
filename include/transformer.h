#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include "config.h"
#include "algebra.h"

typedef struct {
    Matrix *E;

    Matrix *W1q;
    Matrix *W1k;
    Matrix *W1v;

    Matrix *W2q;
    Matrix *W2k;
    Matrix *W2v;
    
    Matrix *W3q;
    Matrix *W3k;
    Matrix *W3v;

    Matrix *W4q;
    Matrix *W4k;
    Matrix *W4v;

    Matrix *Wo;

    Vector *gamma;
    Vector *beta;
} Transformer;

Transformer *transformer_init();
void transformer_free(Transformer *t);

#endif
