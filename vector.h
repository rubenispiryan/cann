#ifndef _VECTOR_HEADER_
#define _VECTOR_HEADER_

#include <stdbool.h>

typedef struct vector Vector;

Vector *create_vector(int n, bool is_column);

void destroy_vector(Vector *v);

int vector_get_n(const Vector *v);

const double *vector_get_data(const Vector *v);

bool vector_get_is_column(const Vector *v);

void vector_transpose(Vector *v);

void vector_set_data(Vector *v, const double *data, int n_elem);

void vector_map_data(Vector *v, double (*const func) (double));

void vector_add(const Vector *v1, const Vector *v2, Vector *dst);

void vector_initialize(Vector *v, double (*const method) (int));

void vector_print(const Vector *v);

#endif