#include <math.h>
#include <assert.h>

#include "activation.h"
#include "vector.h"

static double relu(double input) {
    return input > 0 ? input : 0;
}

static double sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}

void activation_relu(Vector *input) {
    assert(input);
    vector_map_data(input, &relu);
}

void activation_sigmoid(Vector *input) {
    assert(input);
    vector_map_data(input, &sigmoid);
}

void activation_tanh(Vector *input) {
    assert(input);
    vector_map_data(input, &tanh);
}