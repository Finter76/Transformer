#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Vector* vec_init(int size){
    if (size <= 0) return NULL;

    Vector *v = malloc(sizeof(Vector));
    if(!v) return NULL;

    v->data = calloc(size, sizeof(float));
    v->size = size;

    if (v->data == NULL) {
        free(v);
        return NULL;
    }

    return v;
}

Matrix* mat_init(int rows, int cols){
    if (rows <= 0 || cols <= 0) return NULL;
    Matrix *m = malloc(sizeof(Matrix));
    if(!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(float));

    if (m->data == NULL) {
        free(m);
        return NULL;
    }

    return m;
}


void vec_free(Vector* v){
    if(!v || !v->data) return;

    free(v->data);
    free(v);
}

void mat_free(Matrix* m){
    if(!m || !m->data) return;

    free(m->data);
    free(m);
}

Vector* vec_copy(const Vector *a){
    if(!a || !a->data) return NULL;

    Vector *b = malloc(sizeof(Vector));
    if(!b) return NULL;
   
    b->size = a->size;

    b->data = calloc(a->size, sizeof(float));
    if(!b->data) return NULL;

    for(int i = 0; i < a->size; i++){
        b->data[i] = a->data[i];
    }
    
    return b;
}

Matrix* mat_copy(const Matrix *a){
    if(!a || !a->data) return NULL;

    Matrix *b = malloc(sizeof(Matrix));
    if(!b) return NULL;
    
    b->rows = a->rows;
    b->cols = a->cols;

    b->data = calloc(a->rows * a->cols, sizeof(float));
    if(!b->data) return NULL;

    for(int i = 0; i < a->rows * a->cols; i++){
        b->data[i] = a->data[i];
    }
    
    return b;
}

int vec_get(const Vector *v, int i, float *result){
    if(!v || !v->data) return -1;    
    
    if(i < 0 || i >= v->size) return -2;

    *result = v->data[i];

    return 0;
}

int vec_set(Vector* v, int i, float val){
    if(!v || !v->data) return -1;

    if(i < 0 || i >= v->size) return -2;

    v->data[i] = val;

    return 0;
}

int mat_get(const Matrix *m, int row, int col, float* result){
    if(!m || !m->data) return -1;    
 
    if(row < 0 || row >= m->rows) return -2;
    if(col < 0 || col >= m->cols) return -2;

    *result = m->data[row * m->cols + col];

    return 0;
}

int mat_set(Matrix* m, int row, int col, float val){
    if(!m || !m->data) return -1;    
 
    if(row < 0 || row >= m->rows) return -2;
    if(col < 0 || col >= m->cols) return -2;

    m->data[row * m->cols + col] = val;

    return 0;
}

Vector* vec_add(const Vector *a, const Vector *b){
    if(!a || !a->data || !b || !b->data) return NULL;
    if(a->size != b->size) return NULL;

    Vector *c = vec_init(a->size);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] + b->data[i];
    }
    
    return c;
}

Matrix* mat_add(const Matrix* a, const Matrix* b){
    if(!a || !a->data || !b || !b->data) return NULL;
    if(a->rows != b->rows) return NULL;
    if(a->cols != b->cols) return NULL;

    Matrix *c = mat_init(a->rows, a->cols);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->rows * a->cols; i++){
        c->data[i] = a->data[i] + b->data[i];
    }
    
    return c;
}

Vector* vec_sub(const Vector *a, const Vector *b){
    if(!a || !a->data || !b || !b->data) return NULL;
    if(a->size != b->size) return NULL;
    
    Vector *c = vec_init(a->size);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] - b->data[i];
    }
    
    return c;

}

Matrix* mat_sub(const Matrix *a, const Matrix *b){
    if(!a || !a->data || !b || !b->data) return NULL;
    if(a->rows != b->rows) return NULL;
    if(a->cols != b->cols) return NULL;
    
    Matrix *c = mat_init(a->rows, a->cols);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->rows * a->cols; i++){
        c->data[i] = a->data[i] - b->data[i];
    }
    
    return c;
}

Matrix* broadcast_sum(const Matrix *m, const Vector *v){
    if(!m || !m->data || !v || !v->data) return NULL;
    if(v->size != m->cols) return NULL;

    Matrix *result = mat_copy(m);
    if(!result) return NULL;

    for(int i = 0; i < result->rows; i++){
        for(int j = 0; j < result->cols; j++){
            result->data[i * result->cols + j] += v->data[j];
        }
    }

    return result;
}

int dot_prod(const Vector* a, const Vector* b, float *result){
    if(!a || !a->data || !b || !b->data) return -1;

    if(a->size != b->size) return -2;

    *result = 0.0f;

    for(int i = 0; i < a->size; i++){
        *result += a->data[i] * b->data[i];
    }
    
    return 0;
}

Vector* vecmul(const Vector* a, const Vector* b){
    if(!a || !a->data || !b || !b->data) return NULL;

    if(a->size != b->size) return NULL;

    Vector *c = vec_init(a->size);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] * b->data[i];
    }

    return c;
}

Vector* vec_scalar_mul(const Vector* a, float val){
    if(!a || !a->data) return NULL;

    Vector *b = vec_init(a->size);
    if(!b || !b->data) return NULL;

    for(int i = 0; i < a->size; i++){
        b->data[i] = a->data[i] * val;
    }

    return b;
}

Matrix* matmul(const Matrix* a, const Matrix* b){
    if(!a || !a->data || !b || !b->data) return NULL;

    if(a->cols != b->rows) return NULL;

    Matrix *c = mat_init(a->rows, b->cols);
    if(!c || !c->data) return NULL;

    for(int i = 0; i < a->rows; i++){
        for(int j = 0; j < b->cols; j++){
            for(int k = 0; k < a->cols; k++){
                c->data[i * c->cols + j] += 
                    a->data[i * a->cols + k] * 
                    b->data[k * b->cols + j];
            }
        }
    }

    return c;
}

Matrix* mat_scalar_mul(const Matrix* a, float val){
    if(!a || !a->data) return NULL;

    Matrix *b = mat_init(a->rows, a->cols);
    if(!b || !b->data) return NULL;

    for(int i = 0; i < a->rows * a->cols; i++){
        b->data[i] = a->data[i] * val;
    }

    return b;
}

Vector* mat_vec_mul(const Matrix *m, const Vector *v){
    if(!m || !m->data || !v || !v->data) return NULL;

    if(m->cols != v->size) return NULL;

    Vector *y = vec_init(m->rows);
    if(!y || !y->data) return NULL;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            y->data[i] += m->data[i * m->cols + j] * v->data[j];
        }
    }

    return y;
}

Matrix* mat_transpose(const Matrix *a){
    if(!a || !a->data) return NULL;
    
    Matrix *b = mat_init(a->cols, a->rows);
    if(!b || !b->data) return NULL;

    for(int i = 0; i < a->rows; i++){
        for(int j = 0; j < a->cols; j++){
            b->data[j * b->cols + i] = a->data[i * a->cols + j];
        } 
    }

    return b;
}

int vec_norm(const Vector *a, float *result){   
    if(!a || !a->data || !result) return -1;

    *result = 0.0f;
    for(int i = 0; i < a->size; i++){
        *result += a->data[i] * a->data[i];
    }

    *result = sqrt(*result);

    return 0;
}

int vec_sum(const Vector *a, float *result){
    if(!a || !a->data || !result) return -1;

    *result = 0.0f;
    for(int i = 0; i < a->size; i++){
        *result += a->data[i];
    }

    return 0;
}

Vector *mat_get_row(const Matrix* m, int index){
    if(!m || !m->data) return NULL;
    if(index < 0 || index >= m->rows) return NULL;

    Vector *v = vec_init(m->cols);
    if(!v) return NULL;

    for(int i = 0; i < m->cols; i++){
        v->data[i] = m->data[index * m->cols + i];
    } 

    return v;
}

int mat_set_row(Matrix *m, int index, const Vector *v){
    if(!m || !m->data || !v || !v->data) return -1;
    if(index < 0 || index >= m->rows) return -2;
    if (v->size != m->cols) return -3;

    for(int i = 0; i < m->cols; i++){
        m->data[index * m->cols + i] = v->data[i];
    }

    return 0;
}

Matrix* get_submatrix(const Matrix* m, int first_r, int last_r, int first_c, int last_c){
    if(!m || !m->data) return NULL;
    
    if(first_r < 0 || first_r >= last_r || last_r > m->rows)
        return NULL;

    if(first_c < 0 || first_c >= last_c || last_c > m->cols)
        return NULL; 

    int rows = last_r - first_r;
    int cols = last_c - first_c;
 
    Matrix *result = mat_init(rows, cols);
    if(!result) return NULL;

    int k = 0;
    for(int i = first_r; i < last_r; i++){
        for(int j = first_c; j < last_c; j++){
            result->data[k] = m->data[i * m->cols + j];
            k++;
        }
    }

    return result;
}

Matrix* mat_concat_cols(const Matrix *a, const Matrix *b){
    if(!a || !a->data || !b || !b->data) return NULL;
    if(a->rows != b->rows) return NULL;

    Matrix *result = mat_init(a->rows, a->cols + b->cols);
    if(!result) return NULL;

    for(int i = 0; i < result->rows; i++){
        for(int j = 0; j < a->cols; j++){
            result->data[i * result->cols + j] =
                a->data[i * a->cols + j];
        }

        for(int j = 0; j < b->cols; j++){
            result->data[i * result->cols + a->cols + j] =
                b->data[i * b->cols + j];
        }
    }

    return result;
}

Vector* softmax(const Vector *v){
    if(!v || !v->data) return NULL;
    
    float max = v->data[0];
    for(int i = 1; i < v->size; i++){
        if(v->data[i] > max) max = v->data[i];
    }

    Vector *result = vec_copy(v);
    if(!result) return NULL;

    float sum = 0.0f;
    for(int i = 0; i < v->size; i++){
        result->data[i] = expf(result->data[i] - max);
        sum += result->data[i];
    }

    for(int i = 0; i < v->size; i++){
        result->data[i] /= sum;
    }

    return result;
}

Matrix* mat_softmax(const Matrix *m){
    if(!m || !m->data) return NULL;
    
    Matrix *result = mat_init(m->rows, m->cols);
    if(!result) return NULL;

    for(int i = 0; i < m->rows; i++){
        Vector *v = mat_get_row(m, i);
        if(!v){
            mat_free(result);
            return NULL;
        }

        Vector *s = softmax(v);
        vec_free(v);
        if(!s){
            mat_free(result);
            return NULL;
        }

        if(mat_set_row(result, i, s) != 0){
            vec_free(s);
            mat_free(result);
            return NULL;
        }
    
        vec_free(s);
    }

    return result;
}

Vector* reLU(const Vector *v){
    if(!v || !v->data) return NULL;

    Vector *result = vec_copy(v);
    if(!result) return NULL;

    for(int i = 0; i < result->size; i++){
        if(result->data[i] < 0)
            result->data[i] = 0;
    }

    return result;
}

Matrix* mat_reLU(const Matrix *m){
    if(!m || !m->data) return NULL;

    Matrix *result = mat_copy(m);
    if(!result) return NULL;
     
    for(int i = 0; i < result->rows; i++){
        for(int j = 0; j < result->cols; j++){
            if(result->data[i * result->cols + j] < 0)
                result->data[i * result->cols + j] = 0;
        }
    }

    return result;
}

void print_vec(const Vector *v){
    if(!v || !v->data) return;

    printf("[");
    for(int i = 0; i < v->size; i++){
        printf("%.4f", v->data[i]);
        
        if(i < v->size - 1) printf(", ");
    }
    printf("]\n");
}

void print_mat(const Matrix *m){
    if(!m || !m->data) return;

    printf("[\n");

    for(int i = 0; i < m->rows; i++){
        printf("  [");

        for(int j = 0; j < m->cols; j++){
            printf("%.4f", m->data[i * m->cols + j]);

            if(j < m->cols - 1)
                printf(", ");
        }

        printf("]");

        if(i < m->rows - 1)
            printf(",");

        printf("\n");
    }

    printf("]\n");

}
