#include "attention.h"
#include "config.h"
#include <stddef.h>
#include <math.h>

Matrix* attention_head(const Matrix *Q_input, const Matrix *K_input, const Matrix *V_input, const Matrix *Wq, const Matrix *Wk, const Matrix *Wv, int causal){ 
    if(!Q_input || !Q_input->data || !K_input || !K_input->data || !V_input || !V_input->data) return NULL;
    if(!Wq || !Wq->data || !Wk || !Wk->data || !Wv || !Wv->data) return NULL;
    
    if(causal != 0 && causal != 1) return NULL;

    float norm = 1.0f / sqrtf(D_HEAD);
    
    Matrix *Q = matmul(Q_input, Wq);
    if(!Q) goto fatal; 
    Matrix *K = matmul(K_input, Wk);
    if(!K) goto fatal;
    Matrix *V = matmul(V_input, Wv);
    if(!V) goto fatal;

    Matrix *KT = mat_transpose(K);
    if(!KT) goto fatal;

    Matrix *scores = matmul(Q, KT);
    if(!scores) goto fatal;
    Matrix *scaled = mat_scalar_mul(scores, norm);
    if(!scaled) goto fatal;

    if(causal){
        for (int i = 0; i < scaled->rows; i++) {
            for (int j = i + 1; j < scaled->cols; j++) {
                scaled->data[i * scaled->cols + j] = -INFINITY;
            }
        }
    }

    Matrix *A = mat_softmax(scaled);
    if(!A) goto fatal;
    Matrix *H = matmul(A, V);
    if(!H) goto fatal;   

    mat_free(Q); 
    mat_free(K); 
    mat_free(V); 
    mat_free(KT);
    mat_free(scores); 
    mat_free(scaled); 
    mat_free(A);
 
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

Matrix* attention(const Matrix *Q_input, const Matrix *K_input, const Matrix *V_input, Matrix *const Wq[], Matrix *const Wk[], Matrix *const Wv[], const Matrix *Wo, int causal){
    if(!Q_input || !Q_input->data || !K_input || !K_input->data || !V_input || !V_input->data) return NULL;

    if(!Wq || !Wk || !Wv || !Wo || !Wo->data) return NULL;

    if(causal != 0 && causal != 1) return NULL;

    Matrix *heads[NUM_HEADS] = {0};

    for(int i = 0; i < NUM_HEADS; i++){
        if(!Wq[i] || !Wq[i]->data ||
           !Wk[i] || !Wk[i]->data ||
           !Wv[i] || !Wv[i]->data)
            goto fatal;

        heads[i] = attention_head(Q_input, K_input, V_input, Wq[i], Wk[i], Wv[i], causal);

        if(!heads[i]) goto fatal;
    }

    /* Concatenate heads */
    Matrix *H = heads[0];
    heads[0] = NULL;    

    for(int i = 1; i < NUM_HEADS; i++){
        Matrix *tmp = mat_concat_cols(H, heads[i]);

        if(!tmp){
            mat_free(H);
            goto fatal;
        }

        mat_free(H);
        mat_free(heads[i]);

        heads[i] = NULL; 

        H = tmp;
    }

    /* Final linear projection */
    Matrix *output = matmul(H, Wo);

    mat_free(H);

    return output;

fatal:
    for(int i = 0; i < NUM_HEADS; i++)
        mat_free(heads[i]);

    return NULL;
}

