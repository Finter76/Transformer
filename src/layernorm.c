#include "layernorm.h"
#include "config.h"
#include <stddef.h>
#include <math.h>

Matrix* layer_norm(Matrix *m, Transformer *tr){
    if(!m || !m->data) return NULL;

    Matrix *result = mat_init(m->rows, m->cols);

    /* mu_i = 1/d * \sum_{j=1}^{d}{x_{ij}}*/
    Vector *mu = vec_init(m->rows);
    if(!mu) return NULL;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            mu->data[i] += m->data[i * m->cols + j];
        }
        mu->data[i] /= m->cols;
    }

    /* sigma_i = 1/d * \sum_{j=1}^{d}{(x_{ij} - \mu_i)^2} */
    Vector *sigma = vec_init(m->rows);
    if(!sigma) return NULL;
    
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            sigma->data[i] += powf((m->data[i * m->cols + j] - mu->data[i]), 2);
        }
        sigma->data[i] /= m->cols;
    }

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            result->data[i * m->cols + j] = (m->data[i * m->cols + j] - mu->data[i]) / sqrtf(sigma->data[i] + EPSILON);

            result->data[i * m->cols + j] *= tr->gamma->data[j];
            result->data[i * m->cols + j] += tr->beta->data[j];
        }
    }

    return result;
}
