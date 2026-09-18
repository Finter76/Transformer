#include "ffn.h"
#include "config.h"
#include <stddef.h>

// FFN(X) = ReLU(X*W1 + b1)W2 + b2;
Matrix* ffn(const Matrix *X, const Matrix *W1, const Vector *b1, const Matrix *W2, const Vector *b2){    
    if(!X || !X->data ||
       !W1 || !W1->data ||
       !b1 || !b1->data ||
       !W2 || !W2->data ||
       !b2 || !b2->data)
        return NULL;

    Matrix *H = matmul(X, W1);
    if(!H) return NULL;

    Matrix *H_bias = broadcast_sum(H, b1);
    mat_free(H);

    if(!H_bias) return NULL;

    Matrix *A = mat_reLU(H_bias);
    mat_free(H_bias);
    
    if(!A) return NULL;
    
    Matrix *Y = matmul(A, W2);
    mat_free(A);

    if(!Y) return NULL;

    Matrix *result = broadcast_sum(Y, b2);
    mat_free(Y);

    return result;     
}
