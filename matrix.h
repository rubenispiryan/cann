#ifndef _MATRIX_HEADER_
#define _MATRIX_HEADER_

#include "vector.h"

typedef struct matrix Matrix;

Matrix *create_matrix(int n_rows, int n_cols);

void destroy_matrix(Matrix *m);

int matrix_get_n_elem(const Matrix *m);

int matrix_get_n_rows(const Matrix *m);

int matrix_get_n_cols(const Matrix *m);

float *matrix_get_row(const Matrix *m, int row_i);

void matrix_copy(Matrix *dst_m, const Matrix *src_m);

void matrix_set(Matrix *m, float val, int row, int col);

void matrix_vec_mul(const Matrix *m, const Vector *v, Vector *res);

void matrix_T_vec_mul(const Matrix *m, const Vector *v, Vector *res);

void matrix_outer_mul(Matrix *dst, const Vector *left, const Vector *right);

void matrix_scaled_sub(Matrix *dst, const Matrix *m, float scale);

Matrix *matrix_make_from_k(int k, int n_rows, int n_cols);

void matrix_initialize(Matrix *m, float (*const method) (int, int));

void matrix_print(const Matrix *m);

#endif