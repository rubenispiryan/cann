#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>

#include "vector.h"

typedef struct vector {
    double *data;
    int n;
    bool is_column;
} Vector;

Vector *create_vector(int n, bool is_column) {
    Vector *v = malloc(sizeof(Vector));
    if (v == NULL) {
        return NULL;
    }
    double *data = malloc(sizeof(double) * n);
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

int vector_get_n(const Vector *v) {
    assert(v);
    return v->n;
}

const double *vector_get_data(const Vector *v) {
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

void vector_set_data(Vector *v, const double *data, int n_elem) {
    assert(v);
    assert(data);
    assert(n_elem <= v->n);
    memcpy(v->data, data, sizeof(double) * n_elem);
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