#ifndef LLAMA_HLS_TOP_HPP_
#define LLAMA_HLS_TOP_HPP_

// Vitis HLS headers
#ifdef __SYNTHESIS__
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"
#else
#include <cmath>
#include <cstdint>
#endif

#include "tensor.hpp"

namespace swan {

// Main HLS Top-level function for LLaMA inference
// This is the entry point for hardware synthesis
void llama_inference_hls_top(
    // Input/Output memory interfaces
    float* input_embedding,    // Input: token embedding [kDim]
    float* output_logits,      // Output: logits [kVocabSize]
    
    // Model weights in DDR memory
    float* weight_token_embedding,  // [kVocabSize][kDim]
    float* weight_attention_wq,     // [kNumLayers][kDim][kDim]
    float* weight_attention_wk,     // [kNumLayers][kDim][kDim]
    float* weight_attention_wv,     // [kNumLayers][kDim][kDim]
    float* weight_attention_wo,     // [kNumLayers][kDim][kDim]
    float* weight_ffn_w1,          // [kNumLayers][kFFNDim][kDim]
    float* weight_ffn_w2,          // [kNumLayers][kDim][kFFNDim]
    float* weight_ffn_w3,          // [kNumLayers][kFFNDim][kDim]
    float* weight_attention_norm,   // [kNumLayers][kDim]
    float* weight_ffn_norm,        // [kNumLayers][kDim]
    float* weight_final_norm,      // [kDim]
    
    // RoPE precomputed tables
    float* cos_table,          // [kSeqLen][kHeadDim/2]
    float* sin_table,          // [kSeqLen][kHeadDim/2]
    
    // Control parameters
    int position,              // Current position in sequence
    int max_position          // Maximum position to process
);

// Individual optimized kernel functions
void attention_layer_kernel(
    float* input,              // [kDim]
    float* output,             // [kDim]
    float* wq, float* wk, float* wv, float* wo,  // Weight matrices
    float* norm_weights,       // [kDim]
    float* cos_vals, float* sin_vals,  // RoPE values for this position
    int position
);

void ffn_layer_kernel(
    float* input,              // [kDim]
    float* output,             // [kDim]
    float* w1, float* w2, float* w3,  // FFN weight matrices
    float* norm_weights        // [kDim]
);

void rmsnorm_kernel(
    float* input,              // [kDim]
    float* output,             // [kDim]
    float* weights             // [kDim]
);

void rope_kernel(
    float* q_input,            // [kDim]
    float* k_input,            // [kDim]
    float* q_output,           // [kDim]
    float* k_output,           // [kDim]
    float* cos_vals,           // [kHeadDim/2]
    float* sin_vals            // [kHeadDim/2]
);

void matmul_kernel(
    float* input,              // [input_size]
    float* weights,            // [output_size][input_size]
    float* output,             // [output_size]
    int input_size,
    int output_size
);

void softmax_kernel(
    float* input,              // [size]
    float* output,             // [size]
    int size
);

void silu_activation_kernel(
    float* input,              // [size]
    float* output,             // [size]
    int size
);

} // namespace swan

#endif // LLAMA_HLS_TOP_HPP_
