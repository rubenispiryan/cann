#include <stddef.h>

#include "nn.h"
#include "matrix.h"

typedef struct layer {
    Matrix *weights;
    int n;
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
    Layer *l = malloc(sizeof(Layer));
    if (l == NULL) {
        destroy_matrix(weights);
        return NULL;
    }
    l->n = n_output;
    l->weights = weights;
    return l;
}

void destroy_layer(Layer *l) {
    assert(l);
    destroy_matrix(l->weights);
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