#include <stddef.h>
#include "matrix.h"

typedef struct matrix {
    float *data;
    int n_rows;
    int n_cols;
} Matrix;

Matrix *create_matrix(int n_rows, int n_cols) {
    Matrix *m = malloc(sizeof(Matrix));
    if (m == NULL) {
        return NULL;
    }
    m->n_cols = n_cols;
    m->n_rows = n_rows;
    float *data = malloc(sizeof(float) * n_cols * n_rows);
    if (data == NULL) {
        free(m);
        return NULL;
    }
    m->data = data;
    return m;
}

void destroy_matrix(Matrix *m) {
    assert(m);
    free(m->data);
    free(m);
}