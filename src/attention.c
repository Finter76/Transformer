#include "attention.h"

Matrix* attention(Matrix *X, Transformer* tr){
    Matrix *Q1 = matmul(X, tr->W1q);
    if(!Q1) return NULL; 
    Matrix *K1 = matmul(X, tr->W1k);
    if(!K1) return NULL;
    Matrix *V1 = matmul(X, tr->W1v);
    if(!V1) return NULL;
    
    Matrix *Q2 = matmul(X, tr->W2q);
    if(!Q2) return NULL;
    Matrix *K2 = matmul(X, tr->W2k);
    if(!K2) return NULL;
    Matrix *V2 = matmul(X, tr->W2v);
    if(!V2) return NULL;
    
    Matrix *Q3 = matmul(X, tr->W3q);
    if(!Q3) return NULL; 
    Matrix *K3 = matmul(X, tr->W3k);
    if(!K3) return NULL;
    Matrix *V3 = matmul(X, tr->W3v);
    if(!V3) return NULL;
    
    Matrix *Q4 = matmul(X, tr->W4q);
    if(!Q4) return NULL;
    Matrix *K4 = matmul(X, tr->W4k);
    if(!K4) return NULL;
    Matrix *V4 = matmul(X, tr->W4v);
    if(!V4) return NULL;
    
}
