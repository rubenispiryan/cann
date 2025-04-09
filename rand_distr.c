#include <math.h>
#include <stdlib.h>
#include <assert.h> 

#include "rand_distr.h"

int rand_int(int a, int b) {
    assert(a <= b);
    return a + rand() % (b - a + 1);
}

double rand_uniform(double left, double right) {
    assert(left <= right);
    double uniform_number = rand();
    return (right - left) * (uniform_number / (double) RAND_MAX) + left;
}

double rand_normal(double mean, double std) {
    double u1 = rand() / (double) RAND_MAX;
    double u2 = rand() / (double) RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * std + mean;
}

double uniform_xavier(int n_input, int n_output) {
    double range = sqrt(6.0 / (n_input + n_output));
    return rand_uniform(-range, range);
}

double normal_xavier(int n_input, int n_output) {
    double sigma = sqrt(2.0 / (n_input + n_output));
    return rand_normal(0, sigma);
}

double uniform_he(int n_input, int n_output) {
    double left = -sqrt(6.0 / n_input);
    double right = sqrt(6.0 / n_output);
    return rand_uniform(left, right);
}

double normal_he(int n_input, int n_output) {
    double sigma = sqrt(2.0 / n_input);
    return rand_normal(0, sigma);
}