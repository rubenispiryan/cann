#ifndef _ACTIVATION_HEADER_
#define _ACTIVATION_HEADER_

#include "vector.h"

typedef struct activation {
    void (*forward) (Vector *);
    void (*update_delta) (Vector *, const Vector *, const Vector *);
} Activation;

Activation *create_activation(void (*forward) (Vector *),
                              void (*update_delta) (Vector *, const Vector *,
                                                const Vector *));

void destroy_activation(Activation *act);

Activation *make_activation_relu();

Activation *make_activation_sigmoid();

Activation *make_activation_tanh();

Activation *make_activation_softmax();

#endif