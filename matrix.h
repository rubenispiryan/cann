#ifndef _MATRIX_HEADER_
#define _MATRIX_HEADER_

typedef struct matrix Matrix;

Matrix *create_matrix(int n_rows, int n_cols);

void destroy_matrix(Matrix *m);

#endif