#include <math.h>
#include <assert.h>

#include "vector.h"

// TODO: add derivatives
double mse(Vector *pred, Vector *target) {
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

double cross_entropy_binary(Vector *pred, Vector *target) {
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