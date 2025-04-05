#ifndef _NN_HEADER_
#define _NN_HEADER_

typedef struct layer Layer;
typedef struct network Network;

struct layer *create_layer(int n_input, int n_output);

void destroy_layer(Layer *l);

Network *create_network(int n_layers);

void destroy_network(Network *net);

#endif