#include "attention.h" 
#include "config.h"
#include <stddef.h>
#include <math.h>

Matrix* attention_head(const Matrix *X, const Matrix *Wq, const Matrix *Wk, const Matrix *Wv){ 
    if(!X || !X->data || !Wq || !Wq->data || !Wk || !Wk->data || !Wv || !Wv->data)
        return NULL;

    float norm = 1.0f / sqrtf(D_HEAD);
    
    Matrix *Q = matmul(X, Wq);
    if(!Q) goto fatal; 
    Matrix *K = matmul(X, Wk);
    if(!K) goto fatal;
    Matrix *V = matmul(X, Wv);
    if(!V) goto fatal;

    Matrix *KT = mat_transpose(K);
    if(!KT) goto fatal;

    Matrix *scores = matmul(Q, KT);
    if(!scores) goto fatal;
    Matrix *scaled = mat_scalar_mul(scores, norm);
    if(!scaled) goto fatal;
    Matrix *A = mat_softmax(scaled);
    if(!A) goto fatal;
    Matrix *H = matmul(A, V);
    if(!H) goto fatal;   
 
    return H;

fatal:
    mat_free(Q);
    mat_free(K);
    mat_free(V);
    mat_free(KT);
    mat_free(scores);
    mat_free(scaled);
    mat_free(A);

    return NULL;
}

Matrix* attention(const Matrix *X, const EncoderLayer *layer){
    if(!X || !X->data || !layer) return NULL;

    Matrix *heads[NUM_HEADS] = {0};

    for(int i = 0; i < NUM_HEADS; i++){
        heads[i] = attention_head(X, layer->Wq[i], layer->Wk[i], layer->Wv[i]);

        if(!heads[i]){
            for(int j = 0; j < NUM_HEADS; j++)
                mat_free(heads[j]);

            return NULL;
        }
    }

    /* Concatenate heads */
    Matrix *H = heads[0];

    for(int i = 1; i < NUM_HEADS; i++){
        Matrix *tmp = mat_concat_cols(H, heads[i]);

        if(!tmp){
            mat_free(H);

            for(int j = i; j < NUM_HEADS; j++)
                mat_free(heads[j]);

            return NULL;
        }

        mat_free(H);
        mat_free(heads[i]);

        H = tmp;
    }

    /* Final linear projection */
    Matrix *output = matmul(H, layer->Wo);

    mat_free(H);

    return output;
}

