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
    
    // KV Cache for autoregressive generation
    float* k_cache,            // [kNumLayers][kSeqLen][kDim]
    float* v_cache,            // [kNumLayers][kSeqLen][kDim]
    
    // RoPE precomputed tables
    float* cos_table,          // [kSeqLen][kHeadDim/2]
    float* sin_table,          // [kSeqLen][kHeadDim/2]
    
    // Control parameters
    int position,              // Current position in sequence
    int max_position          // Maximum position to process
);

// Enhanced attention layer with KV cache support
void attention_layer_with_cache_kernel(
    float* input,              // [kDim]
    float* output,             // [kDim]
    float* wq, float* wk, float* wv, float* wo,  // Weight matrices
    float* norm_weights,       // [kDim]
    float* k_cache,            // [kSeqLen][kDim] - current layer cache
    float* v_cache,            // [kSeqLen][kDim] - current layer cache
    float* cos_vals, float* sin_vals,  // RoPE values for this position
    int position,
    int max_position
);

// KV Cache management functions
void update_kv_cache_kernel(
    float* k_cache,            // [kSeqLen][kDim]
    float* v_cache,            // [kSeqLen][kDim]
    float* new_k,              // [kDim]
    float* new_v,              // [kDim]
    int position
);

void compute_attention_with_cache_kernel(
    float* q,                  // [kDim] - query for current token
    float* k_cache,            // [kSeqLen][kDim] - all cached keys
    float* v_cache,            // [kSeqLen][kDim] - all cached values
    float* output,             // [kDim] - attention output
    int position              // Current position (0 to position-1 are valid in cache)
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

// External kernel function declarations (implemented in separate kernel_*.cpp files)
extern "C" {
    void kernel_matmul(float* i_vec, float* i_mat, float* o_vec, int vec_size, int col_size);
    void kernel_rmsnorm(float* i_vec_1, float* i_vec_2, float* o_vec, int vec_size);
    void kernel_rope(float* q_in, float* k_in, float* cos_vec, float* sin_vec,
                     float* q_out, float* k_out, int head_begin);
    void kernel_softmax(float* i_vec, float* o_vec, int vec_size);
    void kernel_silu(float* i_vec, float* o_vec, int vec_size);
}

} // namespace swan

#endif // LLAMA_HLS_TOP_HPP_
