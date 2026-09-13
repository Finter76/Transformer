#include "algebra.h"
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
