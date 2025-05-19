#include <arm_neon.h>
#include <assert.h>

#include "simd_neon.h"

void float_add(float* output, const float* input1, const float* input2,
                int len) {
    assert(output);
    assert(input1);
    assert(input2);
    for (int i = 0; i < len; i += 4) {
        float32x4_t v1 = vld1q_f32(&input1[i]);
        float32x4_t v2 = vld1q_f32(&input2[i]);
        float32x4_t res = vaddq_f32(v1, v2);
        vst1q_f32(&output[i], res);
    }
    for (int i = len - (len % 4); i < len; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void float_mul(float* output, const float* input1, const float* input2,
    int len) {
    assert(output);
    assert(input1);
    assert(input2);
    for (int i = 0; i < len; i += 4) {
        float32x4_t v1 = vld1q_f32(&input1[i]);
        float32x4_t v2 = vld1q_f32(&input2[i]);
        float32x4_t res = vmulq_f32(v1, v2);
        vst1q_f32(&output[i], res);
    }
    for (int i = len - (len % 4); i < len; i++) {
        output[i] = input1[i] * input2[i];
    }
}

float float_dot(const float *input1, const float *input2, int len) {
    assert(input1);
    assert(input2);
    float32x4_t total = vdupq_n_f32(0.0f);

    for (int i = 0; i < len; i += 4) {
        float32x4_t v1 = vld1q_f32(&input1[i]);
        float32x4_t v2 = vld1q_f32(&input2[i]);
        float32x4_t res = vmulq_f32(v1, v2);
        total = vmlaq_f32(total, v1, v2);
    }
    float output[4] = {0};
    vst1q_f32(output, total);
    float result = output[0] + output[1] + output[2] + output[3];
    for (int i = len - (len % 4); i < len; i++) {
        result += input1[i] * input2[i];
    }
    return result;
}