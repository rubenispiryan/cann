#ifndef _VECTOR_HEADER_
#define _VECTOR_HEADER_

#include <stdbool.h>

typedef struct vector Vector;

Vector *create_vector(int n, bool is_column);

void destroy_vector(Vector *v);

int vector_get_n(Vector *v);

const float *vector_get_data(Vector *v);

bool vector_get_is_column(Vector *v);

void vector_set_data(Vector *v, float *data);

#endif