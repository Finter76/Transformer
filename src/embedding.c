#include "embedding.h"
#include "config.h"
#include "algebra.h"

Matrix *embed(TokenList *t, Transformer *tr){
    Matrix *X = mat_init(t->words, D_MODEL);
    if(!X) return NULL;

    for(int i = 0; i < t->words; i++){
        int id = t->tokens[i].id;

        Vector *v = mat_get_row(tr->E, id);
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

    return X;
}
