#include <math.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "activation.h"
#include "vector.h"
#include "matrix.h"

Activation *create_activation(void (*forward) (Vector *),
                              void (*update_delta) (Vector *, const Vector *,
                                                const Vector *)) {
    Activation *act = malloc(sizeof(Activation));
    if (act == NULL) {
        return NULL;
    }
    if (forward != NULL) {
        act->forward = forward;
    }
    if (update_delta != NULL) {
        act->update_delta = update_delta;
    }
    return act;
}

void destroy_activation(Activation *act) {
    assert(act);
    free(act);
}

static float relu(float input, float second) {
    return input > 0 ? input : 0;
}

static float relu_dx(float input, float second) {
    return input > 0 ? 1 : 0;
}

static void relu_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &relu, 0);
}

static void relu_backward(Vector *delta, const Vector *input,
                          const Vector *post_act) {
    assert(input);
    assert(delta);
    Vector *act_dx = create_vector(vector_get_n(delta), true);
    assert(act_dx);
    vector_map_data_to(act_dx, input, relu_dx, 0);
    vector_dot(delta, act_dx, delta);
    destroy_vector(act_dx);
}

Activation *make_activation_relu() {
    Activation *act = create_activation(relu_forward, relu_backward);
    return act;
}

static float sigmoid(float input, float second) {
    return 1.0 / (1.0 + exp(-input));
}

static float sigmoid_dx(float input, float second) {
    return input * (1.0 - input);
}

static void sigmoid_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &sigmoid, 0);
}

static void sigmoid_backward(Vector *delta, const Vector *input,
                             const Vector *post_act) {
    assert(input);
    Vector *act_dx = create_vector(vector_get_n(delta), true);
    assert(act_dx);
    vector_map_data_to(act_dx, post_act, sigmoid_dx, 0);
    vector_dot(delta, act_dx, delta);
    destroy_vector(act_dx);
}

Activation *make_activation_sigmoid() {
    Activation *act = create_activation(sigmoid_forward, sigmoid_backward);
    return act;
}

static float ttanh(float input, float second) {
    return tanh(input);
}

static float ttanh_dx(float input, float second) {
    float sech = cosh(input);
    return 1.0 / (sech * sech);
}

static void tanh_forward(Vector *input) {
    assert(input);
    vector_map_data(input, &ttanh, 0);
}

static void tanh_backward(Vector *delta, const Vector *input,
                          const Vector *post_act) {
    assert(input);
    Vector *act_dx = create_vector(vector_get_n(delta), true);
    assert(act_dx);
    vector_map_data_to(act_dx, input, ttanh_dx, 0);
    vector_dot(delta, act_dx, delta);
    destroy_vector(act_dx);
}

Activation *make_activation_tanh() {
    Activation *act = create_activation(tanh_forward, tanh_backward);
    return act;
}

static float exp_with_diff(float input, float second) {
    return exp(input - second);
}

static float divide(float input, float second) {
    return input / second;
}

static void softmax_forward(Vector *input) {
    assert(input);
    int n = vector_get_n(input);
    const float *data = vector_get_data(input);
    float max_val = data[0];
    for (int i = 0; i < n; i++) {
        if (max_val < data[i]) {
            max_val = data[i];
        }
    }
    vector_map_data(input, exp_with_diff, max_val);
    float total = 0;
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
    const float *data = vector_get_data(soft_maxed_data);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            float value = 0;
            if (i == j) {
                value = data[i] * (1.0 - data[i]);
            } else {
                value = -data[i] * data[j];
            }
            matrix_set(jacobian, value, i, j);
        }
    }
}

static void softmax_backward(Vector *delta, const Vector *input,
                             const Vector *post_act) {
    assert(delta);
    assert(input);
    assert(post_act);
    int n = vector_get_n(delta);
    assert(n == vector_get_n(input));
    assert(n == vector_get_n(post_act));
    Matrix *jacobian = create_matrix(n, n);
    Vector *temp = create_vector(n, true);
    assert(temp);
    assert(jacobian);

    softmax_dx(jacobian, post_act);
    matrix_vec_mul(jacobian, delta, temp);
    vector_copy(delta, temp);
    destroy_matrix(jacobian);
    destroy_vector(temp);
}

Activation *make_activation_softmax() {
    return create_activation(softmax_forward, softmax_backward);
}