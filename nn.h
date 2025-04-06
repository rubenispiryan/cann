#ifndef _NN_HEADER_
#define _NN_HEADER_

#include "vector.h"

typedef struct layer Layer;
typedef struct network Network;

Layer *create_layer(int n_input, int n_output);

void destroy_layer(Layer *l);

Network *create_network(int n_layers);

void destroy_network(Network *net);

int net_get_n_output(Network *net);

int layer_get_n_weights(Layer *l);

void layer_set_weights(Layer *l, float *weights, int n_elem);

void net_set_layer(Network *net, Layer *l, int index);

void net_feed_forward(Network *net, Vector *input, Vector *output);

#endif