#include "addnorm.h"
#include "layernorm.h"
#include <stddef.h>

Matrix* add_norm(const Matrix *a, const Matrix *b, const Vector *gamma, const Vector *beta){
    if(!a || !a->data ||
       !b || !b->data ||
       !gamma || !gamma->data ||
       !beta || !beta->data)
        return NULL;

    /* Residual Connection */
    Matrix *residual_output = mat_add(a, b);
    if(!residual_output) return NULL;

    /* Layer Norm */
    Matrix *result = layer_norm(residual_output, gamma, beta);
    mat_free(residual_output);

    return result;
}
