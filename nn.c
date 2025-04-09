#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#include "nn.h"

typedef struct {
    Vector *prev;
    Vector *pre_act;
    Vector *post_act;
    Vector *delta;
} Cache;

typedef struct layer {
    Matrix *weights;
    Vector *bias;
    Vector *output;
    int n;
    Activation *act;
    Cache *cache;
} Layer;

typedef struct network {
    Layer **layers;
    int n_layers;
    Loss *loss;
    double learning_rate;
} Network;

Cache *create_cache(int n_input, int n_output) {
    Cache *cache = malloc(sizeof(Cache));
    if (cache == NULL) {
        return NULL;
    }
    Vector *pre_act = create_vector(n_output, true);
    if (pre_act == NULL) {
        free(cache);
        return NULL;
    }
    Vector *prev = create_vector(n_input, true);
    if (prev == NULL) {
        destroy_vector(pre_act);
        free(cache);
        return NULL;
    }
    Vector *delta = create_vector(n_output, true);
    if (delta == NULL) {
        destroy_vector(prev);
        destroy_vector(pre_act);
        free(cache);
        return NULL;
    }
    Vector *post_act = create_vector(n_output, true);
    if (post_act == NULL) {
        destroy_vector(delta);
        destroy_vector(prev);
        destroy_vector(pre_act);
        free(cache);
        return NULL;
    }
    cache->post_act = post_act;
    cache->pre_act = pre_act;
    cache->delta = delta;
    cache->prev = prev;
    return cache;
}

void destroy_cache(Cache *cache) {
    assert(cache);
    destroy_vector(cache->delta);
    destroy_vector(cache->prev);
    destroy_vector(cache->pre_act);
    destroy_vector(cache->post_act);
    free(cache);
}

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
    Cache *cache = create_cache(n_input, n_output);
    if (cache == NULL) {
        destroy_matrix(weights);
        destroy_vector(bias);
        destroy_vector(output);
        free(l);
        return NULL;
    }
    l->n = n_output;
    l->weights = weights;
    l->bias = bias;
    l->output = output;
    l->act = NULL;
    l->cache = cache;
    return l;
}

void destroy_layer(Layer *l) {
    assert(l);
    destroy_matrix(l->weights);
    destroy_vector(l->bias);
    destroy_vector(l->output);
    destroy_cache(l->cache);
    free(l);
}

Network *create_network(int n_layers, double lr) {
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
    net->learning_rate = lr;
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

void layer_set_activation(Layer *l, Activation *act) {
    assert(l);
    assert(act);
    assert(act->forward);
    assert(act->backward);
    l->act = act;
}

void net_set_loss(Network *net, Loss *loss) {
    assert(net);
    assert(loss);
    assert(loss->forward);
    assert(loss->backward);
    net->loss = loss;
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

static void layer_apply(Layer *l, const Vector *input) {
    assert(l);
    assert(input);
    vector_copy_data(l->cache->prev, input);
    matrix_vec_mul(l->weights, input, l->output);
    vector_add(l->output, l->bias, l->output);
    vector_copy_data(l->cache->pre_act, l->output);
    if (l->act) {
        l->act->forward(l->output);
    }
    vector_copy_data(l->cache->post_act, l->output);
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

double net_forward_loss(const Network *net, const Vector *prediction,
                        const Vector *target) {
    assert(net);
    assert(prediction);
    assert(target);
    assert(net->loss);
    return net->loss->forward(prediction, target);
}

static void layer_update(const Layer *l, const Matrix *dW, const Vector *db,
                        double lr) {
    assert(l);
    assert(dW);
    assert(db);
    matrix_scaled_sub(l->weights, dW, lr);
    vector_scaled_sub(l->bias, db, lr);
}

void net_backpropagation(const Network *net, const Vector *prediciton,
                         const Vector *target) {
    assert(net);
    assert(prediciton);
    assert(target);
    assert(net->loss);

    int n_layers = net->n_layers;
    int n_out = vector_get_n(target);
    assert(n_out == vector_get_n(prediciton));

    Vector *delta = net->layers[n_layers - 1]->cache->delta;
    assert(n_out == vector_get_n(delta));

    net->loss->backward(delta, prediciton, target);
    for (int i = n_layers - 1; i >= 0; i--) {
        Layer *current_layer = net->layers[i];
        Cache *cache = current_layer->cache;
        delta = cache->delta;
        assert(delta);

        Vector *act_dx = create_vector(current_layer->n, true);
        Matrix *dW = create_matrix(current_layer->n,
                                   vector_get_n(cache->prev));

        if (current_layer->act) {
            current_layer->act->backward(act_dx, cache->pre_act,
                                        cache->post_act);
            vector_mul(delta, act_dx, delta);
        }
        vector_transpose(cache->prev);
        matrix_outer_mul(dW, delta, cache->prev);
        vector_transpose(cache->prev);
        if (i > 0) {
            matrix_T_vec_mul(current_layer->weights, delta,
                             net->layers[i - 1]->cache->delta);
        }
        layer_update(current_layer, dW, cache->delta, net->learning_rate);
        destroy_vector(act_dx);
        destroy_matrix(dW);
    }
}