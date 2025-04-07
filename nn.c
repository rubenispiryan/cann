#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>

#include "nn.h"
#include "matrix.h"

// TODO: add backprop
typedef struct layer {
    Matrix *weights;
    Vector *bias;
    Vector *output;
    int n;
    void (*activation) (Vector *);
} Layer;

typedef struct network {
    Layer **layers;
    int n_layers;
} Network;

Layer *create_layer(int n_input, int n_output) {
    Matrix *weights = create_matrix(n_output, n_input);
    if (weights == NULL) {
        return NULL;
    }
    Vector *bias = create_vector(n_output, true);
    if (bias == NULL) {
        destroy_matrix(weights);
        return NULL;
    }
    Vector *output = create_vector(n_output, true);
    if (output == NULL) {
        destroy_matrix(weights);
        destroy_vector(bias);
        return NULL;
    }
    Layer *l = malloc(sizeof(Layer));
    if (l == NULL) {
        destroy_matrix(weights);
        destroy_vector(bias);
        destroy_vector(output);
        return NULL;
    }
    l->n = n_output;
    l->weights = weights;
    l->bias = bias;
    l->output = output;
    l->activation = NULL;
    return l;
}

void destroy_layer(Layer *l) {
    assert(l);
    destroy_matrix(l->weights);
    destroy_vector(l->bias);
    destroy_vector(l->output);
    free(l);
}

Network *create_network(int n_layers) {
    Layer **layers = malloc(sizeof(Layer *) * n_layers);
    if (layers == NULL) {
        return NULL;
    }
    Network *net = malloc(sizeof(Network));
    if (net == NULL) {
        free(layers);
        return NULL;
    }
    net->layers = layers;
    net->n_layers = n_layers;
    return net;
}

static void destroy_network_layers(Network *net) {
    assert(net);
    for (size_t i = 0; i < net->n_layers; i++) {
        destroy_layer(net->layers[i]);
    }
}

void destroy_network(Network *net) {
    assert(net);
    destroy_network_layers(net);
    free(net->layers);
    free(net);
}

static void layer_apply(Layer *l, const Vector *input) {
    assert(l);
    assert(input);
    matrix_vec_mul(l->weights, input, l->output);
    if (l->activation == NULL) {
        return;
    }
    vector_add(l->output, l->bias, l->output);
    l->activation(l->output);
}

int net_get_n_output(const Network *net) {
    assert(net);
    return net->layers[net->n_layers - 1]->n;
}

int layer_get_n_weights(const Layer *l) {
    assert(l);
    return matrix_get_n_elem(l->weights);
}

const Matrix *layer_get_weights(const Layer *l) {
    assert(l);
    return l->weights;
}

void layer_set_weights(const Layer *l, const Matrix *new_weights) {
    assert(l);
    assert(new_weights);
    matrix_copy(l->weights, new_weights);
}

void layer_set_activation(Layer *l, void (*func) (Vector *)) {
    assert(l);
    assert(func);
    l->activation = func;
}

void layer_initialize_weights(const Layer *l,
                              double (*const method) (int, int)) {
    assert(l);
    assert(method);
    matrix_initialize(l->weights, method);
}

static double zero_initializator(int n) {
    return 0;
}

void layer_initialize_bias(const Layer *l) {
    assert(l);
    vector_initialize(l->bias, &zero_initializator);
}

void layer_initialize(const Layer *l, double (*const method) (int, int)) {
    assert(l);
    layer_initialize_weights(l, method);
    layer_initialize_bias(l);
}

void net_set_layer(Network *net, Layer *l, int index) {
    assert(net);
    assert(l);
    assert(index < net->n_layers);
    assert(index >= 0);
    net->layers[index] = l;
}

void net_feed_forward(const Network *net, const Vector *input, Vector *output) {
    assert(net);
    assert(input);
    assert(output);
    for (int i = 0; i < net->n_layers; i++) {
        Layer *current_layer = net->layers[i];
        layer_apply(current_layer, input);
        input = current_layer->output;
    }
    assert(vector_get_n(output) == vector_get_n(input));
    vector_set_data(output, vector_get_data(input), vector_get_n(input));
}