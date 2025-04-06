#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#include "matrix.h"

typedef struct matrix {
    double *data;
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
    double *data = malloc(sizeof(double) * n_cols * n_rows);
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

int matrix_get_n_elem(const Matrix *m) {
    assert(m);
    return m->n_cols * m->n_rows;
}

void matrix_copy(Matrix *dst_m, const Matrix *src_m) {
    assert(dst_m);
    assert(src_m);
    assert(dst_m->n_cols == src_m->n_cols);
    assert(dst_m->n_rows == src_m->n_rows);
    memcpy(dst_m->data, src_m->data, sizeof(double) * dst_m->n_cols * dst_m->n_rows);
}

void matrix_vec_mul(const Matrix *m, const Vector *v, Vector *res) {
    assert(m);
    assert(v);
    assert(res);

    assert(v != res);
    assert(m->n_cols == vector_get_n(v));
    assert(m->n_rows == vector_get_n(res));
    assert(vector_get_is_column(v));
    assert(vector_get_is_column(res));

    const double *data = vector_get_data(v);
    double *res_data = malloc(sizeof(double) * vector_get_n(res));
    if (res_data == NULL) {
        return;
    }
    for (int i = 0; i < m->n_rows; i++) {
        double total = 0;
        for (int j = 0; j < m->n_cols; j++) {
            total += m->data[i * m->n_cols + j] * data[j];
        }
        res_data[i] = total;
    }
    vector_set_data(res, res_data, vector_get_n(res));
}

Matrix *matrix_make_from_k(int k, int n_rows, int n_cols) {
    Matrix *m = create_matrix(n_rows, n_cols);
    if (m == NULL) {
        return NULL;
    }
    for (int i = 0; i < n_rows * n_cols; i++) {
        m->data[i] = k;
    }
    return m;
}

void matrix_initialize(Matrix *m, double (*const method) (int, int)) {
    assert(m);
    assert(method);
    for (int i = 0; i < m->n_cols * m->n_rows; i++) {
        m->data[i] = method(m->n_cols, m->n_rows);
    }
}

void matrix_print(const Matrix *m) {
    assert(m);
    for (int i = 0; i < m->n_rows; i++) {
        for (int j = 0; j < m->n_cols; j++) {
            if (j > 0) {
                printf(" ");
            }
            printf("%.2f", m->data[i * m->n_cols + j]);
        }
        printf("\n");
    }
}