#include "transformer.h"
#include "algebra.h"
#include "config.h"
#include <stdlib.h>

Transformer* transformer_init(void){
    Transformer *t = malloc(sizeof(Transformer));
    if(!t) return NULL;

    /* Embedding */
    t->E_encoder = mat_init(VOCAB_SIZE, D_MODEL);
    if(!t->E_encoder) goto fail;

    /* ==================== */
    /* Encoder Layers       */
    /* ==================== */

    for(int i = 0; i < ENCODER_LAYERS; i++){
        EncoderLayer *layer = &t->encoder_layers[i];

        /* Multi-Head Self-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            layer->Wq[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wk[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wv[j] = mat_init(D_MODEL, D_HEAD);

            if(!layer->Wq[j] || !layer->Wk[j] || !layer->Wv[j]) 
                goto fail;
        }

        layer->Wo = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo) goto fail;

        /* First LayerNorm */
        layer->gamma1 = vec_init(D_MODEL);
        layer->beta1  = vec_init(D_MODEL);

        if(!layer->gamma1 || !layer->beta1) goto fail;

        /* FFN */
        layer->W1 = mat_init(D_MODEL, D_FF);
        layer->b1 = vec_init(D_FF);

        layer->W2 = mat_init(D_FF, D_MODEL);
        layer->b2 = vec_init(D_MODEL);

        if(!layer->W1 || !layer->b1 || !layer->W2 || !layer->b2)
            goto fail;

        /* Second LayerNorm */
        layer->gamma2 = vec_init(D_MODEL);
        layer->beta2  = vec_init(D_MODEL);

        if(!layer->gamma2 || !layer->beta2) goto fail;
    }

    /* ==================== */
    /* Decoder Layers       */
    /* ==================== */

    for(int i = 0; i < DECODER_LAYERS; i++){

        DecoderLayer *layer = &t->decoder_layers[i];

        /* Masked Self-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            layer->Wq_self[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wk_self[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wv_self[j] = mat_init(D_MODEL, D_HEAD);

            if(!layer->Wq_self[j] || !layer->Wk_self[j] || !layer->Wv_self[j])
                goto fail;
        }

        layer->Wo_self = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo_self) goto fail;

        /* First LayerNorm */
        layer->gamma1 = vec_init(D_MODEL);
        layer->beta1  = vec_init(D_MODEL);

        if(!layer->gamma1 || !layer->beta1) goto fail;

        /* Cross-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            layer->Wq_cross[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wk_cross[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wv_cross[j] = mat_init(D_MODEL, D_HEAD);

            if(!layer->Wq_cross[j] || !layer->Wk_cross[j] || !layer->Wv_cross[j])
                goto fail;
        }

        layer->Wo_cross = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo_cross) goto fail;

        /* Second LayerNorm */
        layer->gamma2 = vec_init(D_MODEL);
        layer->beta2  = vec_init(D_MODEL);

        if(!layer->gamma2 || !layer->beta2) goto fail;

        /* FFN */
        layer->W1 = mat_init(D_MODEL, D_FF);
        layer->b1 = vec_init(D_FF);

        layer->W2 = mat_init(D_FF, D_MODEL);
        layer->b2 = vec_init(D_MODEL);

        if(!layer->W1 || !layer->b1 || !layer->W2 || !layer->b2)
            goto fail;

        /* Third LayerNorm */
        layer->gamma3 = vec_init(D_MODEL);
        layer->beta3  = vec_init(D_MODEL);

        if(!layer->gamma3 || !layer->beta3) goto fail;
    }

    return t;

fail:
    transformer_free(t);
    return NULL;
}


void transformer_free(Transformer *t){
    if(!t) return;

    mat_free(t->E_encoder);

    /* ==================== */
    /* Encoder Layers       */
    /* ==================== */

    for(int i = 0; i < ENCODER_LAYERS; i++){
        EncoderLayer *layer = &t->encoder_layers[i];

        for(int j = 0; j < NUM_HEADS; j++){
            mat_free(layer->Wq[j]);
            mat_free(layer->Wk[j]);
            mat_free(layer->Wv[j]);
        }

        mat_free(layer->Wo);

        vec_free(layer->gamma1);
        vec_free(layer->beta1);

        mat_free(layer->W1);
        vec_free(layer->b1);

        mat_free(layer->W2);
        vec_free(layer->b2);

        vec_free(layer->gamma2);
        vec_free(layer->beta2);
    }

    /* ==================== */
    /* Decoder Layers       */
    /* ==================== */

    for(int i = 0; i < DECODER_LAYERS; i++){

        DecoderLayer *layer = &t->decoder_layers[i];

        /* Masked Self-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            mat_free(layer->Wq_self[j]);
            mat_free(layer->Wk_self[j]);
            mat_free(layer->Wv_self[j]);
        }

        mat_free(layer->Wo_self);

        vec_free(layer->gamma1);
        vec_free(layer->beta1);

        /* Cross-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            mat_free(layer->Wq_cross[j]);
            mat_free(layer->Wk_cross[j]);
            mat_free(layer->Wv_cross[j]);
        }

        mat_free(layer->Wo_cross);

        vec_free(layer->gamma2);
        vec_free(layer->beta2);

        /* FFN */
        mat_free(layer->W1);
        vec_free(layer->b1);

        mat_free(layer->W2);
        vec_free(layer->b2);

        vec_free(layer->gamma3);
        vec_free(layer->beta3);
    }

    free(t);
}
