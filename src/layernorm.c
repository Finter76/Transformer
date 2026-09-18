#include "layernorm.h"
#include "config.h"
#include <stddef.h>
#include <math.h>

Matrix* layer_norm(const Matrix *m, const Vector *gamma, const Vector *beta){
    if(!m || !m->data || !gamma || !gamma->data || !beta || !beta->data) return NULL;

     if(gamma->size != m->cols || beta->size != m->cols) return NULL;

    Matrix *result = mat_init(m->rows, m->cols);
    if(!result) return NULL;

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
            float diff = m->data[i * m->cols + j] - mu->data[i];
            sigma->data[i] += (diff * diff);
        }
        sigma->data[i] /= m->cols;
    }

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            int idx = i * m->cols + j;

            result->data[idx] = (m->data[idx] - mu->data[i]) / sqrtf(sigma->data[i] + EPSILON);

            result->data[idx] *= gamma->data[j];
            result->data[idx] += beta->data[j];
        }
    }
    
    vec_free(mu);
    vec_free(sigma);

    return result;
}
