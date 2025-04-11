#include <math.h>
#include <stdlib.h>
#include <assert.h> 

#include "rand_distr.h"

int rand_int(int a, int b) {
    assert(a <= b);
    return a + rand() % (b - a + 1);
}

float rand_uniform(float left, float right) {
    assert(left <= right);
    float uniform_number = rand();
    return (right - left) * (uniform_number / (float) RAND_MAX) + left;
}

float rand_normal(float mean, float std) {
    float u1 = rand() / (float) RAND_MAX;
    float u2 = rand() / (float) RAND_MAX;
    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * std + mean;
}

float uniform_xavier(int n_input, int n_output) {
    float range = sqrt(6.0 / (n_input + n_output));
    return rand_uniform(-range, range);
}

float normal_xavier(int n_input, int n_output) {
    float sigma = sqrt(2.0 / (n_input + n_output));
    return rand_normal(0, sigma);
}

float uniform_he(int n_input, int n_output) {
    float left = -sqrt(6.0 / n_input);
    float right = sqrt(6.0 / n_output);
    return rand_uniform(left, right);
}

float normal_he(int n_input, int n_output) {
    float sigma = sqrt(2.0 / n_input);
    return rand_normal(0, sigma);
}

void shuffle(int *data, int n) {
    assert(data);
    for (int i = 0; i < n; i++) {
        int idx = rand_int(i, n - 1);
        int temp = data[i];
        data[i] = data[idx];
        data[idx] = temp;
    }
}