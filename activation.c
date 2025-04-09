#include <math.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "activation.h"
#include "vector.h"
#include "matrix.h"

Activation *create_activation(void (*forward) (Vector *),
                              void (*backward) (Vector *, const Vector *,
                                                const Vector *)) {
    Activation *act = malloc(sizeof(Activation));
    if (act == NULL) {
        return NULL;
    }
    if (forward != NULL) {
        act->forward = forward;
    }
    if (backward != NULL) {
        act->backward = backward;
    }
    return act;
}

void destroy_activation(Activation *act) {
    assert(act);
    free(act);
}

static double relu(double input, double second) {
    return input > 0 ? input : 0;
}

static double relu_dx(double input, double second) {
    return input > 0 ? 1 : 0;
}

static void relu_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &relu, 0);
}

static void relu_backward(Vector *output, const Vector *input,
                          const Vector *post_act) {
    assert(input);
    assert(output);
    vector_map_data_to(output, input, relu_dx, 0);
}

Activation *make_activation_relu() {
    Activation *act = create_activation(relu_forward, relu_backward);
    return act;
}

static double sigmoid(double input, double second) {
    return 1.0 / (1.0 + exp(-input));
}

static double sigmoid_dx(double input, double second) {
    return input * (1.0 - input);
}

static void sigmoid_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &sigmoid, 0);
}

static void sigmoid_backward(Vector *output, const Vector *input,
                             const Vector *post_act) {
    assert(input);
    assert(output);
    vector_map_data_to(output, post_act, sigmoid_dx, 0);
}

Activation *make_activation_sigmoid() {
    Activation *act = create_activation(sigmoid_forward, sigmoid_backward);
    return act;
}

static double ttanh(double input, double second) {
    return tanh(input);
}

static double ttanh_dx(double input, double second) {
    double sech = cosh(input);
    return 1.0 / (sech * sech);
}

static void tanh_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &ttanh, 0);
}

static void tanh_backward(Vector *output, const Vector *input,
                          const Vector *post_act) {
    assert(input);
    assert(output);
    vector_map_data_to(output, input, ttanh_dx, 0);
}

Activation *make_activation_tanh() {
    Activation *act = create_activation(tanh_forward, tanh_backward);
    return act;
}

static double exp_with_diff(double input, double second) {
    return exp(input - second);
}

static double divide(double input, double second) {
    return input / second;
}

static void softmax_forward(Vector *input) {
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

static void softmax_dx(Matrix *jacobian, const Vector *soft_maxed_data) {
    assert(soft_maxed_data);
    assert(jacobian);
    int n_rows = matrix_get_n_rows(jacobian);
    int n_cols = matrix_get_n_cols(jacobian);
    const double *data = vector_get_data(soft_maxed_data);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            double value = 0;
            if (i == j) {
                value = data[i] * (1.0 - data[i]);
            } else {
                value = -data[i] * data[j];
            }
            matrix_set(jacobian, value, i, j);
        }
    }
}

static void softmax_backward(Vector *output, const Vector *input,
                             const Vector *post_act) {
    assert(output);
    assert(input);
    assert(post_act);
    int n = vector_get_n(output);
    assert(n == vector_get_n(input));
    assert(n == vector_get_n(post_act));
    Matrix *jacobian = create_matrix(n, n);
    softmax_dx(jacobian, post_act);
}
// TODO: implement
Activation *make_activation_softmax() {
    assert(false);
    return NULL;
}