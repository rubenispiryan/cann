#ifndef _SIMD_HEADER_
#define _SIMD_HEADER_

void float_add(float *output, const float *input1,
               const float *input2, int len);

void float_mul(float *output, const float *input1,
               const float *input2, int len);

float float_dot(const float *input1, const float *input2, int len);

#endif