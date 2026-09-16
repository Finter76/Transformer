#include "attention.h"
#include "config.h"
#include <stddef.h>
#include <math.h>

Matrix* attention_head(Matrix *X, Matrix *Wq, Matrix *Wk, Matrix *Wv){ 
    if(!X || !Wq || !Wk || !Wv)
        return NULL;

    float norm = 1.0f / sqrtf(D_HEAD);
    
    Matrix *Q = matmul(X, Wq);
    if(!Q) return NULL; 
    Matrix *K = matmul(X, Wk);
    if(!K) return NULL;
    Matrix *V = matmul(X, Wv);
    if(!V) return NULL;

    Matrix *KT = mat_transpose(K);
    if(!KT) return NULL;

    Matrix *scores = matmul(Q, KT);
    Matrix *scaled = mat_scalar_mul(scores, norm);
    Matrix *A = mat_softmax(scaled);
    Matrix *H = matmul(A, V);
    
    return H;
}

Matrix* attention(Matrix *X, Transformer* tr){
    Matrix *H1 = attention_head(X, tr->W1q, tr->W1k, tr->W1v); 
    if(!H1) return NULL;

    Matrix *H2 = attention_head(X, tr->W2q, tr->W2k, tr->W2v);
    if(!H2) return NULL;

    Matrix *H3 = attention_head(X, tr->W3q, tr->W3k, tr->W3v);
    if(!H3) return NULL;

    Matrix *H4 = attention_head(X, tr->W4q, tr->W4k, tr->W4v);
    if(!H4) return NULL;

    Matrix *H12 = mat_concat_cols(H1, H2);
    if(!H12) return NULL;
    Matrix *H34 = mat_concat_cols(H3, H4);
    if(!H34) return NULL;
    Matrix *H = mat_concat_cols(H12, H34);
    if(!H) return NULL;    

    Matrix *output = matmul(H, tr->Wo);
    if(!output) return NULL;

    return output;    
}
