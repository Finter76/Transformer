#include "positional.h"
#include <math.h>
#include <stddef.h>

Matrix* positional_encoding(Matrix* X){
    Matrix *PE = mat_init(X->rows, X->cols);
    if(!PE) return NULL;

    for(int i = 0; i < X->rows; i++){
        for(int j = 0; j < X->cols; j++){
            float val = 0.0f;

            int pair = j / 2; 
            float exponent = 2.0f * pair / D_MODEL;
            float denominator = powf(10000.0f, exponent);
            
            if(j % 2 == 0) val = sinf(i/denominator);
            else val = cosf(i/denominator);

            if(mat_set(PE, i, j, val) != 0) return NULL;
        }
    }

    Matrix *result = mat_add(X, PE);
    mat_free(PE);

    return result;
}
