#include "embedding.h"
#include "config.h"
#include <math.h>

Matrix *embed(const int *ids, int length, const Matrix *E){
    Matrix *X = mat_init(length, D_MODEL);
    if(!X) return NULL;

    for(int i = 0; i < length; i++){
        int id = ids[i];

        Vector *v = mat_get_row(E, id);
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
