#ifndef TENSOR_FPGA_HPP_
#define TENSOR_FPGA_HPP_

// Vitis HLS headers - only include when synthesizing
#ifdef __SYNTHESIS__
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"
#else
// Fallback headers for simulation/testbench
#include <cmath>
#include <cstdint>
#endif

#include "tensor.hpp"

// Vitis HLS optimized functions for LLaMA2 100M
// Scalar operations
void AddFPGA(float* out, const float* in, float a, int size);
void MulFPGA(float* out, const float* in, float a, int size);

// Element-wise operations
void AddFPGA(float* out, const float* lhs, const float* rhs, int size);
void MulFPGA(float* out, const float* lhs, const float* rhs, int size);

// Matrix multiplication for different tensor types
void MatmulFPGA(float* out, const float* in, const float* w, 
                int in_size, int out_size);

// Specialized matrix multiplication for attention
void MatmulAttnFPGA(float* out, const float* in, const float* w,
                    int dim);

// Specialized matrix multiplication for FFN
void MatmulFFNFPGA(float* out, const float* in, const float* w,
                   int in_dim, int out_dim);

// Normalization
void RMSNormFPGA(float* out, const float* in, const float* w, int size);

// Attention operations
void SoftmaxFPGA(float* out, const float* in, int size, int max_pos);

// Rotary Position Embedding
void RoPEFPGA(float* q_out, float* k_out, const float* q_in,
              const float* k_in, const float* cos_vec,
              const float* sin_vec, int head_begin, int head_size);

// Activation functions
void SiLUFPGA(float* out, const float* in, int size);



void VocabProjectionFPGA(float* logits, const float* hidden, 
                        const float* embedding_weights, 
                        int vocab_size, int hidden_dim);

#endif // TENSOR_FPGA_HPP_
