#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include "config.h"
#include "algebra.h"

typedef struct {
    Matrix *Wq[NUM_HEADS];
    Matrix *Wk[NUM_HEADS];
    Matrix *Wv[NUM_HEADS];

    Matrix *Wo;

    Vector *gamma1;
    Vector *beta1;

    Matrix *W1;
    Vector *b1;

    Matrix *W2;
    Vector *b2;

    Vector *gamma2;
    Vector *beta2;
    
} EncoderLayer;

typedef struct {
    /* Masked Self-Attention */
    Matrix *Wq_self[NUM_HEADS];
    Matrix *Wk_self[NUM_HEADS];
    Matrix *Wv_self[NUM_HEADS];
    Matrix *Wo_self;

    Vector *gamma1;
    Vector *beta1;

    /* Cross-Attention */
    Matrix *Wq_cross[NUM_HEADS];
    Matrix *Wk_cross[NUM_HEADS];
    Matrix *Wv_cross[NUM_HEADS];
    Matrix *Wo_cross;

    Vector *gamma2;
    Vector *beta2;

    /* FFN */
    Matrix *W1;
    Vector *b1;
    Matrix *W2;
    Vector *b2;

    Vector *gamma3;
    Vector *beta3;

} DecoderLayer;

typedef struct {
    Matrix *E_encoder;
    Matrix *E_decoder;
    
    EncoderLayer encoder_layers[ENCODER_LAYERS];
    DecoderLayer decoder_layers[DECODER_LAYERS];

} Transformer;

Transformer *transformer_init();
void transformer_free(Transformer *t);

#endif
