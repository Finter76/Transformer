#include "transformer.h"
#include "algebra.h"
#include "config.h"
#include <stdlib.h>

Transformer* transformer_init(){
    Transformer* t = malloc(sizeof(Transformer));
    t->E = mat_init(VOCAB_SIZE, D_MODEL);

    t->W1q = mat_init(VOCAB_SIZE, D_MODEL);
    t->W1k = mat_init(VOCAB_SIZE, D_MODEL);
    t->W1v = mat_init(VOCAB_SIZE, D_MODEL);
    
    return t;
}
void transformer_free(Transformer *t){
    mat_free(t->E);

    free(t);
}
