#include "embedding.h"
#include "config.h"
#include <math.h>

Matrix *embed(TokenList *t, Transformer *tr){
    Matrix *X = mat_init(t->words, D_MODEL);
    if(!X) return NULL;

    for(int i = 0; i < t->words; i++){
        int id = t->tokens[i].id;

        Vector *v = mat_get_row(tr->E_encoder, id);
        if(!v){
            mat_free(X);
            return NULL;
        }
    
        if(mat_set_row(X, i, v) != 0){
            vec_free(v);    
            mat_free(X);
            return NULL;
        } 

        vec_free(v);
    }

    Matrix *scaled = mat_scalar_mul(X, sqrtf(D_MODEL));
    mat_free(X);

    return scaled; 
}
