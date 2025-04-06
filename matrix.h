#ifndef _MATRIX_HEADER_
#define _MATRIX_HEADER_

#include "vector.h"

typedef struct matrix Matrix;

Matrix *create_matrix(int n_rows, int n_cols);

void destroy_matrix(Matrix *m);

int matrix_get_n_elem(const Matrix *m);

void matrix_copy(Matrix *dst_m, const Matrix *src_m);

void matrix_vec_mul(const Matrix *m, const Vector *v, Vector *res);

Matrix *matrix_make_from_k(int k, int n_rows, int n_cols);

void matrix_initialize(Matrix *m, double (*const method) (int, int));

void matrix_print(const Matrix *m);

#endif