#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>

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

int matrix_get_n_elem(Matrix *m) {
    assert(m);
    return m->n_cols * m->n_rows;
}

void matrix_set_data(Matrix *m, float *data, int n_elem) {
    assert(m);
    assert(data);
    assert(n_elem <= m->n_cols * m->n_rows);
    memcpy(m->data, data, sizeof(float) * n_elem);
}

void matrix_vec_mul(Matrix *m, Vector *v, Vector *res) {
    assert(m);
    assert(v);
    assert(res);

    assert(v != res);
    assert(m->n_cols == vector_get_n(v));
    assert(m->n_rows == vector_get_n(res));
    assert(vector_get_is_column(v));
    assert(vector_get_is_column(res));

    const float *data = vector_get_data(v);
    float *res_data = malloc(sizeof(float) * vector_get_n(res));
    if (res_data == NULL) {
        return;
    }
    for (int i = 0; i < m->n_rows; i++) {
        int total = 0;
        for (int j = 0; j < m->n_cols; j++) {
            total += m->data[i * m->n_cols + j] * data[j];
        }
        res_data[i] = total;
    }
    vector_set_data(res, res_data, vector_get_n(res));
}