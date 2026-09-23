#include "transformer.h"
#include "algebra.h"
#include "config.h"
#include <stdlib.h>
#include <math.h>

Transformer* transformer_init(void){
    Transformer *t = malloc(sizeof(Transformer));
    if(!t) return NULL;

    /* Embedding */
    t->E_encoder = mat_init(VOCAB_SIZE_IT, D_MODEL);
    if(!t->E_encoder) goto fail;
    mat_init_normal(t->E_encoder, 0.0f, 0.02f);   
 
    t->E_decoder = mat_init(VOCAB_SIZE_EN, D_MODEL);
    if(!t->E_decoder) goto fail;
    mat_init_normal(t->E_decoder, 0.0f, 0.02f);

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

            if(!layer->Wq[j] || !layer->Wk[j] || !layer->Wv[j]) goto fail;

            mat_init_xavier(layer->Wq[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wk[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wv[j], D_MODEL, D_HEAD);
        }

        layer->Wo = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo) goto fail;
        mat_init_xavier(layer->Wo, D_MODEL, D_MODEL);

        /* First LayerNorm */
        layer->gamma1 = vec_init(D_MODEL);
        layer->beta1  = vec_init(D_MODEL);
        vec_init_const(layer->gamma1, 1.0f);
        if(!layer->gamma1 || !layer->beta1) goto fail;

        /* FFN */
        layer->W1 = mat_init(D_MODEL, D_FF);
        layer->b1 = vec_init(D_FF);

        layer->W2 = mat_init(D_FF, D_MODEL);
        layer->b2 = vec_init(D_MODEL);

        if(!layer->W1 || !layer->b1 || !layer->W2 || !layer->b2) goto fail;

        mat_init_normal(layer->W1, 0.0f, sqrtf(2.0f / D_MODEL));  
        mat_init_xavier(layer->W2, D_FF, D_MODEL);
    
        /* Second LayerNorm */
        layer->gamma2 = vec_init(D_MODEL);
        layer->beta2  = vec_init(D_MODEL);

        if(!layer->gamma2 || !layer->beta2) goto fail;
        vec_init_const(layer->gamma2, 1.0f);
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

            if(!layer->Wq_self[j] || !layer->Wk_self[j] || !layer->Wv_self[j]) goto fail;

            mat_init_xavier(layer->Wq_self[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wk_self[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wv_self[j], D_MODEL, D_HEAD);
        }

        layer->Wo_self = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo_self) goto fail;
        mat_init_xavier(layer->Wo_self, D_MODEL, D_MODEL);

        /* First LayerNorm */
        layer->gamma1 = vec_init(D_MODEL);
        layer->beta1  = vec_init(D_MODEL);

        if(!layer->gamma1 || !layer->beta1) goto fail;
        vec_init_const(layer->gamma1, 1.0f);

        /* Cross-Attention */
        for(int j = 0; j < NUM_HEADS; j++){
            layer->Wq_cross[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wk_cross[j] = mat_init(D_MODEL, D_HEAD);
            layer->Wv_cross[j] = mat_init(D_MODEL, D_HEAD);

            if(!layer->Wq_cross[j] || !layer->Wk_cross[j] || !layer->Wv_cross[j]) goto fail;

            mat_init_xavier(layer->Wq_cross[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wk_cross[j], D_MODEL, D_HEAD);
            mat_init_xavier(layer->Wv_cross[j], D_MODEL, D_HEAD);
        }

        layer->Wo_cross = mat_init(D_MODEL, D_MODEL);
        if(!layer->Wo_cross) goto fail;
        mat_init_xavier(layer->Wo_cross, D_MODEL, D_MODEL);

        /* Second LayerNorm */
        layer->gamma2 = vec_init(D_MODEL);
        layer->beta2  = vec_init(D_MODEL);

        if(!layer->gamma2 || !layer->beta2) goto fail;
        
        vec_init_const(layer->gamma2, 1.0f);

        /* FFN */
        layer->W1 = mat_init(D_MODEL, D_FF);
        layer->b1 = vec_init(D_FF);

        layer->W2 = mat_init(D_FF, D_MODEL);
        layer->b2 = vec_init(D_MODEL);

        if(!layer->W1 || !layer->b1 || !layer->W2 || !layer->b2) goto fail;

        mat_init_normal(layer->W1, 0.0f, sqrtf(2.0f / D_MODEL));
        mat_init_xavier(layer->W2, D_FF, D_MODEL);
        
        /* Third LayerNorm */
        layer->gamma3 = vec_init(D_MODEL);
        layer->beta3  = vec_init(D_MODEL);

        if(!layer->gamma3 || !layer->beta3) goto fail;

        vec_init_const(layer->gamma3, 1.0f);
    }

    t->W_out = mat_init(D_MODEL, VOCAB_SIZE_EN);
    if(!t->W_out) goto fail;
    mat_init_xavier(t->W_out, D_MODEL, VOCAB_SIZE_EN);

    return t;

fail:
    transformer_free(t);
    return NULL;
}


void transformer_free(Transformer *t){
    if(!t) return;

    mat_free(t->E_encoder);
    mat_free(t->E_decoder);

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
    
        /* Output Projection */
        mat_free(t->W_out);

    free(t);
}
