#ifndef _NN_HEADER_
#define _NN_HEADER_

#include "vector.h"
#include "matrix.h"
#include "loss.h"
#include "activation.h"

typedef struct layer Layer;
typedef struct network Network;

Layer *create_layer(int n_input, int n_output);

void destroy_layer(Layer *l);

Network *create_network(int n_layers, float lr);

void destroy_network(Network *net);

int net_get_n_output(const Network *net);

int layer_get_n_weights(const Layer *l);

const Matrix *layer_get_weights(const Layer *l);

void layer_set_weights(const Layer *l, const Matrix *new_weights);

void layer_set_activation(Layer *l, Activation *act);

void net_set_loss(Network *net, Loss *loss);

void layer_initialize_weights(const Layer *l, float (*const method) (int, int));

void layer_initialize_bias(const Layer *l);

void layer_initialize(const Layer *l, float (*const method) (int, int));

void net_set_layer(Network *net, Layer *l, int index);

void net_predict(const Network *net, const Vector *input, Vector *output);

float net_forward_loss(const Network *net, const Vector *prediciton,
                        const Vector *target);

void net_backpropagation(const Network *net, const Vector *prediciton,
                         const Vector *target);

void net_predict_batch(const Network *net, const Matrix *X, Matrix *Y_hat);

float net_loss_batch(const Network *net, const Matrix *Y_hat, const Matrix *Y);

void net_train(const Network *net, const Matrix *X, const Matrix *Y,
               int epochs);
#endif