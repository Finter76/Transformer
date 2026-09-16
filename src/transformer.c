#include "transformer.h"
#include "algebra.h"
#include "config.h"
#include <stdlib.h>

Transformer* transformer_init(){
    Transformer* t = malloc(sizeof(Transformer));
    if(!t) return NULL;

    t->E = mat_init(VOCAB_SIZE, D_MODEL);

    t->W1q = mat_init(D_MODEL, D_HEAD);
    t->W1k = mat_init(D_MODEL, D_HEAD);
    t->W1v = mat_init(D_MODEL, D_HEAD);
    
    t->W2q = mat_init(D_MODEL, D_HEAD);
    t->W2k = mat_init(D_MODEL, D_HEAD);
    t->W2v = mat_init(D_MODEL, D_HEAD);
    
    t->W3q = mat_init(D_MODEL, D_HEAD);
    t->W3k = mat_init(D_MODEL, D_HEAD);
    t->W3v = mat_init(D_MODEL, D_HEAD);
    
    t->W4q = mat_init(D_MODEL, D_HEAD);
    t->W4k = mat_init(D_MODEL, D_HEAD);
    t->W4v = mat_init(D_MODEL, D_HEAD);

    t->Wo = mat_init(D_MODEL, D_MODEL);

    t->gamma = vec_init(D_MODEL);
    t->beta = vec_init(D_MODEL);
    return t;
}
void transformer_free(Transformer *t){
    if(!t) return;

    mat_free(t->E);

    mat_free(t->W1q);
    mat_free(t->W1k);
    mat_free(t->W1v);

    mat_free(t->W2q);
    mat_free(t->W2k);
    mat_free(t->W2v);

    mat_free(t->W3q);
    mat_free(t->W3k);
    mat_free(t->W3v);

    mat_free(t->W4q);
    mat_free(t->W4k);
    mat_free(t->W4v);

    mat_free(t->Wo);

    free(t);
}
