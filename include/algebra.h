#ifndef ALGEBRA_H_
#define ALGEBRA_H_

typedef struct {
    float *data;
    int size;
} Vector;

typedef struct{
    float *data;
    int rows;
    int cols;
} Matrix;

Vector* vec_init(int);
Matrix* mat_init(int, int);

Vector* vec_copy(const Vector *);
Matrix* mat_copy(const Matrix *);

void vec_free(Vector*);
void mat_free(Matrix*);

int vec_get(const Vector*, int, float*);
int vec_set(Vector*, int, float);

int mat_get(const Matrix*, int, int, float*);
int mat_set(Matrix*, int, int, float);

Vector* vec_add(const Vector*, const Vector*);
Matrix* mat_add(const Matrix*, const Matrix*);

Vector* vec_sub(const Vector*, const Vector*);
Matrix* mat_sub(const Matrix*, const Matrix*);

int dot_prod(const Vector*, const Vector*, float*);
Vector* vecmul(const Vector*, const Vector*);
Vector* vec_scalar_mul(const Vector*, float);

Matrix* matmul(const Matrix*, const Matrix*);
Matrix* mat_scalar_mul(const Matrix*, float);

Vector* mat_vec_mul(const Matrix*, const Vector*);

Matrix* mat_transpose(const Matrix*);

int vec_norm(const Vector*, float*);
int vec_sum(const Vector*, float*);

Vector *mat_get_row(Matrix*, int);
int mat_set_row(Matrix*, int, const Vector*);

void print_vec(const Vector*);
void print_mat(const Matrix*);
#endif
