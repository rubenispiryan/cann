#include <math.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "loss.h"
#include "vector.h"

Loss *create_loss(double (*forward) (const Vector *, const Vector *),
                  void (*backward) (Vector *, const Vector *, const Vector *)) {
    Loss *l = malloc(sizeof(Loss));
    if (l == NULL) {
        return NULL;
    }
    if (forward != NULL) {
        l->forward = forward;
    }
    if (backward != NULL) {
        l->backward = backward;
    }
    return l;
}

void destroy_loss(Loss *l) {
    assert(l);
    free(l);
}

static double mse_forward(const Vector *pred, const Vector *target) {
    assert(pred);
    assert(target);
    int n = vector_get_n(pred);
    assert(n == vector_get_n(target));
    assert(vector_get_is_column(pred) == vector_get_is_column(target));
    double total = 0;
    const double *pred_data = vector_get_data(pred);
    const double *target_data = vector_get_data(target);
    for (int i = 0; i < n; i++) {
        double diff = pred_data[i] - target_data[i];
        total += diff * diff;
    }
    total /= n;
    return total;
}

static void mse_backward(Vector *delta, const Vector *pred, 
                         const Vector *target) {
    assert(delta);
    assert(pred);
    assert(target);
    int n = vector_get_n(pred);
    assert(n == vector_get_n(target));
    assert(n == vector_get_n(delta));
    const double *pred_data = vector_get_data(pred);
    const double *target_data = vector_get_data(target);
    for (int i = 0; i < n; i++) {
        vector_set(delta, (2.0 / n) * (pred_data[i] - target_data[i]), i);
    }
}

Loss *make_mse() {
    Loss *l = create_loss(mse_forward, mse_backward);
    return l;
}

// TODO: implement
static double cross_entropy_binary(Vector *pred, Vector *target) {
    assert(pred);
    assert(target);
    int n = vector_get_n(pred);
    assert(n == vector_get_n(target));
    assert(vector_get_is_column(pred) == vector_get_is_column(target));
    double total = 0;
    const double *pred_data = vector_get_data(pred);
    const double *target_data = vector_get_data(target);
    const double epsilon = 1e-12;
    for (int i = 0; i < n; i++) {
        double p = fmax(fmin(pred_data[i], 1.0 - epsilon), epsilon);
        total += target_data[i] * log(p) + (1 - target_data[i]) * log(1 - p);
    }
    total /= n;
    return -total;
}