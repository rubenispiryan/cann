#ifndef _NN_HEADER_
#define _NN_HEADER_

#include "vector.h"
#include "matrix.h"

typedef struct layer Layer;
typedef struct network Network;

Layer *create_layer(int n_input, int n_output);

void destroy_layer(Layer *l);

Network *create_network(int n_layers);

void destroy_network(Network *net);

int net_get_n_output(const Network *net);

int layer_get_n_weights(const Layer *l);

const Matrix *layer_get_weights(const Layer *l);

void layer_set_weights(const Layer *l, const Matrix *new_weights);

void layer_set_activation(Layer *l, void (*func) (Vector *));

void layer_initialize_weights(const Layer *l, double (*const method) (int, int));

void net_set_layer(Network *net, Layer *l, int index);

void net_feed_forward(const Network *net, const Vector *input, Vector *output);

#endif