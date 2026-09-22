#include "encoder.h"
#include "config.h"
#include "attention.h"
#include "ffn.h"
#include "addnorm.h"
#include <stddef.h>

Matrix* decoder_layer(const Matrix *X, const Matrix *encoder_output, const DecoderLayer *layer){
    if(!X || !X->data || !encoder_output || !encoder_output->data || !layer)
        return NULL;
    
    /* Masked Multi-Head Self-Attention */
    Matrix *self_attn_output = attention(X, X, X, layer->Wq_self, layer->Wk_self, layer->Wv_self, layer->Wo_self, 1);
    if(!self_attn_output) return NULL;

    /* Add & Norm */
    Matrix *Y = add_norm(X, self_attn_output, layer->gamma1, layer->beta1);

    mat_free(self_attn_output);

    if(!Y) return NULL;

    /* Cross Attention */
    Matrix *cross_attn_output = attention(Y, encoder_output, encoder_output, layer->Wq_cross, layer->Wk_cross, layer->Wv_cross, layer->Wo_cross, 0);
    
    if(!cross_attn_output){
        mat_free(Y);
        return NULL;
    }

    /* Add & Norm */
    Matrix *Z = add_norm(
        Y,
        cross_attn_output,
        layer->gamma2,
        layer->beta2
    );

    mat_free(Y);
    mat_free(cross_attn_output);

    if(!Z) return NULL;

    /* FFN */
    Matrix *F = ffn(Z, layer->W1, layer->b1, layer->W2, layer->b2);
    if(!F){
        mat_free(Z);
        return NULL;
    }

    /* Add & Norm */
    Matrix *output = add_norm(Z, F, layer->gamma3, layer->beta3);

    mat_free(Z);
    mat_free(F);

    return output;
}

Matrix* decoder_forward(const Matrix *X, const Matrix *encoder_output, const Transformer *tr){
    if(!X || !X->data || !encoder_output || !encoder_output->data || !tr) return NULL;

    Matrix *current = mat_copy(X);
    if(!current) return NULL;

    for(int i = 0; i < DECODER_LAYERS; i++){
        Matrix *next = decoder_layer(current, encoder_output, &tr->decoder_layers[i]);

        mat_free(current);

        if(!next) return NULL;

        current = next;
    }

    return current;
}
