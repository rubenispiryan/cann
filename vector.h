#ifndef _VECTOR_HEADER_
#define _VECTOR_HEADER_

#include <stdbool.h>

typedef struct vector Vector;

Vector *create_vector(int n, bool is_column);

void destroy_vector(Vector *v);

void vector_copy_data(Vector *dst, const Vector *src);

int vector_get_n(const Vector *v);

float vector_get(const Vector *v, int index);

const float *vector_get_data(const Vector *v);

bool vector_get_is_column(const Vector *v);

void vector_transpose(Vector *v);

void vector_set_data(Vector *v, const float *data, int n_elem);

void vector_set(Vector *v, float val, int index);

void vector_map_data(Vector *v, float (*const func) (float, float),
                     float second);

void vector_map_data_to(Vector *dst, const Vector *src,
                        float (*const func) (float, float),
                        float second);

void vector_add(const Vector *v1, const Vector *v2, Vector *dst);

void vector_dot(const Vector *v1, const Vector *v2, Vector *dst);

void vector_scaled_sub(Vector *dst, const Vector *v, float scale);

void vector_initialize(Vector *v, float (*const method) (int));

void vector_print(const Vector *v);

#endif