#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>

#include "vector.h"
#include "simd_neon.h"

typedef struct vector {
    float *data;
    int n;
    bool is_column;
} Vector;

Vector *create_vector(int n, bool is_column) {
    Vector *v = malloc(sizeof(Vector));
    if (v == NULL) {
        return NULL;
    }
    float *data = NULL;
    if (posix_memalign((void **) &data, 16, sizeof(float) * n) != 0 ||
        data == NULL) {
        free(v);
        return NULL;
    }
    v->data = data;
    v->n = n;
    v->is_column = is_column;
    return v;
}

void destroy_vector(Vector *v) {
    assert(v);
    assert(v->data);
    free(v->data);
    free(v);
}

void vector_free_data(Vector *v) {
    assert(v);
    assert(v->data);
    free(v->data);
}

void vector_copy(Vector *dst, const Vector *src) {
    assert(dst);
    assert(src);
    assert(dst->n == src->n);
    assert(dst->is_column == src->is_column);
    memcpy(dst->data, src->data, src->n * sizeof(float));
}

int vector_get_n(const Vector *v) {
    assert(v);
    return v->n;
}

float vector_get(const Vector *v, int index) {
    assert(v);
    assert(v->n > index && index >= 0);
    return v->data[index];
}

const float *vector_get_data(const Vector *v) {
    assert(v);
    return v->data;
}

bool vector_get_is_column(const Vector *v) {
    assert(v);
    return v->is_column;
}

void vector_transpose(Vector *v) {
    assert(v);
    v->is_column = !v->is_column;
}

void vector_copy_data(Vector *v, const float *data, int n_elem) {
    assert(v);
    assert(data);
    assert(n_elem == v->n);
    memcpy(v->data, data, sizeof(float) * n_elem);
}

void vector_set_data(Vector *v, float *data, int n_elem) {
    assert(v);
    assert(data);
    assert(n_elem == v->n);
    v->data = data;
}

void vector_set(Vector *v, float val, int index) {
    assert(v);
    assert(v->n > index && index >= 0);
    v->data[index] = val;
}

void vector_map_data(Vector *v, float (*const func) (float, float),
                     float second) {
    assert(v);
    assert(func);
    for (int i = 0; i < v->n; i++) {
        v->data[i] = func(v->data[i], second);
    }
}

void vector_map_data_to(Vector *dst, const Vector *src,
                        float (*const func) (float, float),
                        float second) {
    assert(dst);
    assert(src);
    assert(func);
    assert(dst->n == src->n);
    assert(dst->is_column == src->is_column);
    for (int i = 0; i < src->n; i++) {
        dst->data[i] = func(src->data[i], second);
    }
}

void vector_add(const Vector *v1, const Vector *v2, Vector *dst) { 
    assert(v1);
    assert(v2);
    assert(v1->n == v2->n);
    assert(v1->n == dst->n);
    assert(v1->is_column == v2->is_column);
    assert(v1->is_column == dst->is_column);
    float_add(dst->data, v1->data, v2->data, v1->n);
}

void vector_dot(const Vector *v1, const Vector *v2, Vector *dst) {
    assert(v1);
    assert(v2);
    assert(dst);
    assert(v1->n == v2->n);
    assert(v1->n == dst->n);
    assert(v1->is_column == v2->is_column);
    assert(v1->is_column == dst->is_column);
    float_mul(dst->data, v1->data, v2->data, v1->n);
}

void vector_scaled_sub(Vector *dst, const Vector *v, float scale) {
    assert(dst);
    assert(v);
    assert(dst->n == v->n);
    assert(dst->is_column == dst->is_column);
    for (int i = 0; i < dst->n; i++) {
        dst->data[i] -= scale * v->data[i];
    }
}

void vector_initialize(Vector *v, float (*const method) (int)) {
    assert(v);
    assert(method);
    for (int i = 0; i < v->n; i++) {
        v->data[i] = method(v->n);
    }
}

void vector_print(const Vector *v) {
    assert(v);
    for (int i = 0; i < v->n; i++) {
        if (i > 0 && v->is_column) {
            printf("\n");
        } else if (i > 0) {
            printf(" ");
        }
        printf("%.2f", v->data[i]);
    }
    printf("\n");
}