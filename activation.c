#include <math.h>
#include <assert.h>

#include "activation.h"
#include "vector.h"

// TODO: add derivatives for backprop
static double relu(double input, double second) {
    return input > 0 ? input : 0;
}

static double sigmoid(double input, double second) {
    return 1.0 / (1.0 + exp(-input));
}

static double ttanh(double input, double second) {
    return tanh(input);
}

static double exp_with_diff(double input, double second) {
    return exp(input - second);
}

static double divide(double input, double second) {
    return input / second;
}

void activation_relu(Vector *input) {
    assert(input);
    vector_map_data(input, &relu, 0);
}

void activation_sigmoid(Vector *input) {
    assert(input);
    vector_map_data(input, &sigmoid, 0);
}

void activation_tanh(Vector *input) {
    assert(input);
    vector_map_data(input, &ttanh, 0);
}

void activation_softmax(Vector *input) {
    assert(input);
    int n = vector_get_n(input);
    const double *data = vector_get_data(input);
    double max_val = data[0];
    for (int i = 0; i < n; i++) {
        if (max_val < data[i]) {
            max_val = data[i];
        }
    }
    vector_map_data(input, exp_with_diff, max_val);
    double total = 0;
    for (int i = 0; i < n; i++) {
        total += data[i];
    }
    vector_map_data(input, divide, total);
}