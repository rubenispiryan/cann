#ifndef _RAND_DISTR_
#define _RAND_DISTR_

int rand_int(int a, int b);

float rand_uniform(float left, float right);

float rand_normal(float mean, float std);

float uniform_xavier(int n_input, int n_output);

float normal_xavier(int n_input, int n_output);

float uniform_he(int n_input, int n_output);

float normal_he(int n_input, int n_output);

void shuffle(int *data, int n);

#endif