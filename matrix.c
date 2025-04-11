#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

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

int matrix_get_n_elem(const Matrix *m) {
    assert(m);
    return m->n_cols * m->n_rows;
}

int matrix_get_n_rows(const Matrix *m) {
    assert(m);
    return m->n_rows;
}

int matrix_get_n_cols(const Matrix *m) {
    assert(m);
    return m->n_cols;
}

float *matrix_get_row(const Matrix *m, int row_i) {
    assert(m);
    assert(m->n_rows > row_i && row_i >= 0);
    return &m->data[row_i * m->n_cols];
}

void matrix_copy(Matrix *dst_m, const Matrix *src_m) {
    assert(dst_m);
    assert(src_m);
    assert(dst_m->n_cols == src_m->n_cols);
    assert(dst_m->n_rows == src_m->n_rows);
    memcpy(dst_m->data, src_m->data, sizeof(float) * dst_m->n_cols * dst_m->n_rows);
}

void matrix_set(Matrix *m, float val, int row, int col) {
    assert(m);
    assert(m->n_rows > row && row >= 0);
    assert(m->n_cols > col && col >= 0);
    m->data[row * m->n_cols + col] = val;
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

    const float *data = vector_get_data(v);
    float *res_data = malloc(sizeof(float) * vector_get_n(res));
    if (res_data == NULL) {
        return;
    }
    for (int i = 0; i < m->n_rows; i++) {
        float total = 0;
        for (int j = 0; j < m->n_cols; j++) {
            total += m->data[i * m->n_cols + j] * data[j];
        }
        res_data[i] = total;
    }
    vector_set_data(res, res_data, vector_get_n(res));
}

void matrix_T_vec_mul(const Matrix *m, const Vector *v, Vector *res) {
    assert(m);
    assert(v);
    assert(res);

    assert(v != res);
    assert(m->n_cols == vector_get_n(res));
    assert(m->n_rows == vector_get_n(v));
    assert(vector_get_is_column(v));
    assert(vector_get_is_column(res));

    const float *data = vector_get_data(v);
    float *res_data = malloc(sizeof(float) * vector_get_n(res));
    if (res_data == NULL) {
        return;
    }
    for (int i = 0; i < m->n_cols; i++) {
        float total = 0;
        for (int j = 0; j < m->n_rows; j++) {
            total += m->data[j * m->n_cols + i] * data[j];
        }
        res_data[i] = total;
    }
    vector_set_data(res, res_data, vector_get_n(res));
}

void matrix_outer_mul(Matrix *dst, const Vector *left, const Vector *right) {
    assert(dst);
    assert(left);
    assert(right);
    assert(left != right);
    assert(dst->n_cols == vector_get_n(right));
    assert(dst->n_rows == vector_get_n(left));
    assert(vector_get_is_column(left));
    assert(!vector_get_is_column(right));

    const float *left_data = vector_get_data(left);
    const float *right_data = vector_get_data(right);
    for (int i = 0; i < dst->n_rows; i++) {
        for (int j = 0; j < dst->n_cols; j++) {
            dst->data[i * dst->n_cols + j] = left_data[i] * right_data[j];
        }
    }
}

void matrix_scaled_sub(Matrix *dst, const Matrix *m, float scale) {
    assert(dst);
    assert(m);
    int cols = dst->n_cols;
    int rows = dst->n_rows;
    assert(cols == m->n_cols);
    assert(rows == m->n_rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst->data[i * cols + j] -= scale * m->data[i * cols + j];
        }
    }
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

void matrix_initialize(Matrix *m, float (*const method) (int, int)) {
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