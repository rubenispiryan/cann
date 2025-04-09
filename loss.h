#ifndef _LOSS_HEADER_
#define _LOSS_HEADER_

#include "vector.h"

typedef struct loss {
    double (*forward) (const Vector *, const Vector *);
    void (*backward) (Vector *, const Vector *, const Vector *);
} Loss;

Loss *create_loss(double (*forward) (const Vector *, const Vector *),
                  void (*backward) (Vector *, const Vector *, const Vector *));

void destroy_loss(Loss *l);

Loss *make_mse();

Loss *make_cross_entropy_binary();

#endif