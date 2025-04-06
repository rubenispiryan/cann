#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include <memory.h>
#include <stdlib.h>

#include "vector.h"

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
    float *data = malloc(sizeof(float) * n);
    if (data == NULL) {
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
    free(v->data);
    free(v);
}

int vector_get_n(Vector *v) {
    assert(v);
    return v->n;
}

const float *vector_get_data(Vector *v) {
    assert(v);
    return v->data;
}

bool vector_get_is_column(Vector *v) {
    assert(v);
    return v->is_column;
}

void vector_set_data(Vector *v, const float *data, int n_elem) {
    assert(v);
    assert(data);
    assert(n_elem <= v->n);
    memcpy(v->data, data, sizeof(float) * n_elem);
}