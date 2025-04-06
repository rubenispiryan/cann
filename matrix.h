#ifndef _MATRIX_HEADER_
#define _MATRIX_HEADER_

#include "vector.h"

typedef struct matrix Matrix;

Matrix *create_matrix(int n_rows, int n_cols);

void destroy_matrix(Matrix *m);

int matrix_get_n_elem(Matrix *m);

void matrix_set_data(Matrix *m, float *data, int n_elem);

void matrix_vec_mul(Matrix *m, Vector *v, Vector *res);

#endif