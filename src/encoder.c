#include "encoder.h"
#include "config.h"
#include "attention.h"
#include "ffn.h"
#include "addnorm.h"
#include <stddef.h>

Matrix* encoder_layer(const Matrix *X, const EncoderLayer *layer){
    if(!X || !X->data || !layer) return NULL;

    /* Multi-Head Self-Attention */
    Matrix *attn_output = attention(X, X, X, layer->Wq, layer->Wk, layer->Wv, layer->Wo, 0);
    if(!attn_output) return NULL;

    /* Add & Norm */
    Matrix *Y = add_norm(X, attn_output, layer->gamma1, layer->beta1);

    mat_free(attn_output);

    if(!Y) return NULL;

    /* FFN */
    Matrix *F = ffn(Y, layer->W1, layer->b1, layer->W2, layer->b2);
    if(!F){
        mat_free(Y);
        return NULL;
    }

    /* Add & Norm */
    Matrix *Z = add_norm(Y, F, layer->gamma2, layer->beta2);

    mat_free(Y);
    mat_free(F);

    return Z;
}

Matrix* encoder_forward(const Matrix *X, const Transformer *tr){
    if(!X || !X->data || !tr) return NULL;

    Matrix *current = mat_copy(X);
    if(!current) return NULL;

    for(int i = 0; i < ENCODER_LAYERS; i++){
        Matrix *next = encoder_layer(current, &tr->encoder_layers[i]);

        mat_free(current);

        if(!next) return NULL;

        current = next;
    }

    return current;
}
