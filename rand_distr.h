#ifndef _RAND_DISTR_
#define _RAND_DISTR_

int rand_int(int a, int b);

double rand_uniform(double left, double right);

double rand_normal(double mean, double std);

double uniform_xavier(int n_input, int n_output);

double normal_xavier(int n_input, int n_output);

double uniform_he(int n_input, int n_output);

double normal_he(int n_input, int n_output);

#endif