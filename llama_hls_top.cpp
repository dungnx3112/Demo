#include "llama_hls_top.hpp"

// External kernel function declarations (implemented in separate files)
extern "C" {
    void kernel_matmul(float* i_vec, float* i_mat, float* o_vec, int vec_size, int col_size);
    void kernel_rmsnorm(float* i_vec_1, float* i_vec_2, float* o_vec, int vec_size);
    void kernel_rope(float* q_in, float* k_in, float* cos_vec, float* sin_vec,
                     float* q_out, float* k_out, int head_begin);
    void kernel_softmax(float* i_vec, float* o_vec, int vec_size);
    void kernel_silu(float* i_vec, float* o_vec, int vec_size);
}

// Additional HLS headers for math functions
#ifdef __SYNTHESIS__
#include "hls_math.h"
#else
#include <cmath>
// For simulation, map HLS functions to standard math
namespace hls {
    inline float sqrtf(float x) { return ::sqrtf(x); }
    inline float expf(float x) { return ::expf(x); }
    inline float cosf(float x) { return ::cosf(x); }
    inline float sinf(float x) { return ::sinf(x); }
}
#endif

namespace swan {

// Main HLS Top-level function - optimized for autoregressive inference with KV cache
void llama_inference_hls_top(
    // Input/Output memory interfaces
    float* input_embedding,    
    float* output_logits,      
    
    // Model weights in DDR memory
    float* weight_token_embedding,
    float* weight_attention_wq,
    float* weight_attention_wk,
    float* weight_attention_wv,
    float* weight_attention_wo,
    float* weight_ffn_w1,
    float* weight_ffn_w2,
    float* weight_ffn_w3,
    float* weight_attention_norm,
    float* weight_ffn_norm,
    float* weight_final_norm,
    
    // KV Cache for autoregressive generation
    float* k_cache,
    float* v_cache,
    
    // RoPE precomputed tables
    float* cos_table,
    float* sin_table,
    
    // Control parameters
    int position,
    int max_position
) {
    // HLS Interface pragmas for AXI - Enhanced with KV cache
#pragma HLS INTERFACE m_axi port=input_embedding offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output_logits offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=weight_token_embedding offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=weight_attention_wq offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=weight_attention_wk offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=weight_attention_wv offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=weight_attention_wo offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=weight_ffn_w1 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=weight_ffn_w2 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=weight_ffn_w3 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=weight_attention_norm offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=weight_ffn_norm offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=weight_final_norm offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=k_cache offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=v_cache offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=cos_table offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=sin_table offset=slave bundle=gmem16

#pragma HLS INTERFACE s_axilite port=position bundle=control
#pragma HLS INTERFACE s_axilite port=max_position bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Use smaller, dynamically managed buffers to avoid stack overflow
    // Process in streaming fashion where possible
    static float hidden_state[kDim];
    static float attn_output[kDim];
    static float ffn_output[kDim];
    static float temp_buffer[kDim];
    
    // Partition arrays for better memory access patterns - REDUCE PARTITIONING
#pragma HLS ARRAY_PARTITION variable=hidden_state cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=attn_output cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=ffn_output cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_buffer cyclic factor=4 dim=1

    // Load input embedding into local buffer
    load_input_loop: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        hidden_state[i] = input_embedding[i];
    }

    // Process through all transformer layers - REDUCE COMPLEXITY
    transformer_layers: for (int layer = 0; layer < 2; layer++) {  // Only 2 layers for testing
#pragma HLS PIPELINE OFF
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
        
        // Calculate weight offsets for current layer
        int layer_offset = layer * kDim;
        int attn_offset = layer * kDim * kDim;
        int ffn_w1_offset = layer * kFFNDim * kDim;
        int ffn_w2_offset = layer * kDim * kFFNDim;
        int cache_layer_offset = layer * kSeqLen * kDim;
        
        // Get RoPE values for current position
        float cos_vals[kHeadDim/2];
        float sin_vals[kHeadDim/2];
#pragma HLS ARRAY_PARTITION variable=cos_vals complete dim=1
#pragma HLS ARRAY_PARTITION variable=sin_vals complete dim=1
        
        rope_load: for (int i = 0; i < kHeadDim/2; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
            cos_vals[i] = cos_table[position * (kHeadDim/2) + i];
            sin_vals[i] = sin_table[position * (kHeadDim/2) + i];
        }
        
        // Attention layer with KV cache
        attention_layer_with_cache_kernel(
            hidden_state,
            attn_output,
            &weight_attention_wq[attn_offset],
            &weight_attention_wk[attn_offset],
            &weight_attention_wv[attn_offset],
            &weight_attention_wo[attn_offset],
            &weight_attention_norm[layer_offset],
            &k_cache[cache_layer_offset],
            &v_cache[cache_layer_offset],
            cos_vals,
            sin_vals,
            position,
            max_position
        );
        
        // Residual connection after attention
        residual_attn: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II=1
            hidden_state[i] = hidden_state[i] + attn_output[i];
        }
        
        // FFN layer
        ffn_layer_kernel(
            hidden_state,
            ffn_output,
            &weight_ffn_w1[ffn_w1_offset],
            &weight_ffn_w2[ffn_w2_offset],
            &weight_ffn_w3[ffn_w1_offset],
            &weight_ffn_norm[layer_offset]
        );
        
        // Residual connection after FFN
        residual_ffn: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
            hidden_state[i] = hidden_state[i] + ffn_output[i];
        }
    }
    
    // Final layer normalization
    rmsnorm_kernel(hidden_state, temp_buffer, weight_final_norm);
    
    // Copy normalized result back
    copy_norm: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        hidden_state[i] = temp_buffer[i];
    }
    
    // Compute output logits (matrix multiplication with token embedding weights)
    compute_logits: for (int i = 0; i < kVocabSize; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS LOOP_TRIPCOUNT min=32000 max=32000
        float sum = 0.0f;
        dot_product: for (int j = 0; j < kDim; j++) {
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
            sum += hidden_state[j] * weight_token_embedding[i * kDim + j];
        }
        output_logits[i] = sum;
    }
}

// Enhanced Attention layer implementation with KV cache
void attention_layer_with_cache_kernel(
    float* input,
    float* output,
    float* wq, float* wk, float* wv, float* wo,
    float* norm_weights,
    float* k_cache,            // [kSeqLen][kDim]
    float* v_cache,            // [kSeqLen][kDim]
    float* cos_vals, float* sin_vals,
    int position,
    int max_position
) {
#pragma HLS INLINE OFF
    
    // Use smaller buffers with reduced partitioning to save resources
    static float normed_input[kDim];
    static float q[kDim], k[kDim], v[kDim];
    static float q_rope[kDim], k_rope[kDim];
    
    // Conservative array partitioning for resource efficiency - REDUCE FACTORS
#pragma HLS ARRAY_PARTITION variable=normed_input cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q_rope cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=k_rope cyclic factor=4 dim=1
    
    // STAGE 1: Pre-normalization
    rmsnorm_kernel(input, normed_input, norm_weights);
    
    // STAGE 2: Q,K,V projections (sequential to avoid dataflow conflicts)
    matmul_kernel(normed_input, wq, q, kDim, kDim);
    matmul_kernel(normed_input, wk, k, kDim, kDim);
    matmul_kernel(normed_input, wv, v, kDim, kDim);
    
    // STAGE 3: RoPE rotation
    rope_kernel(q, k, q_rope, k_rope, cos_vals, sin_vals);
    
    // STAGE 4: Update KV cache
    update_kv_cache_kernel(k_cache, v_cache, k_rope, v, position);
    
    // STAGE 5: Compute attention with full cache
    static float attn_output[kDim];
#pragma HLS ARRAY_PARTITION variable=attn_output cyclic factor=4 dim=1
    
    compute_attention_with_cache_kernel(q_rope, k_cache, v_cache, attn_output, position + 1);
    
    // STAGE 6: Output projection
    matmul_kernel(attn_output, wo, output, kDim, kDim);
}

// KV Cache update kernel
void update_kv_cache_kernel(
    float* k_cache,            // [kSeqLen][kDim]
    float* v_cache,            // [kSeqLen][kDim]
    float* new_k,              // [kDim]
    float* new_v,              // [kDim]
    int position
) {
#pragma HLS INLINE OFF
    
    int cache_offset = position * kDim;
    
    // Update K cache
    update_k: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        k_cache[cache_offset + i] = new_k[i];
    }
    
    // Update V cache
    update_v: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        v_cache[cache_offset + i] = new_v[i];
    }
}

// Compute attention with full KV cache
void compute_attention_with_cache_kernel(
    float* q,                  // [kDim] - query for current token
    float* k_cache,            // [kSeqLen][kDim] - all cached keys
    float* v_cache,            // [kSeqLen][kDim] - all cached values
    float* output,             // [kDim] - attention output
    int seq_len               // Current sequence length (positions 0 to seq_len-1 are valid)
) {
#pragma HLS INLINE OFF
    
    const float scale = 1.0f / hls::sqrtf((float)kHeadDim);
    
    // Multi-head attention computation
    static float q_heads[kNumHeads][kHeadDim];
    static float attn_output_heads[kNumHeads][kHeadDim];
    static float attn_scores[kSeqLen];  // Attention scores for current head
    
#pragma HLS ARRAY_PARTITION variable=q_heads complete dim=1
#pragma HLS ARRAY_PARTITION variable=attn_output_heads complete dim=1
#pragma HLS ARRAY_PARTITION variable=attn_scores cyclic factor=4 dim=1
    
    // Split query into heads
    split_q_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS UNROLL
        int head_start = head * kHeadDim;
        for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            q_heads[head][i] = q[head_start + i];
        }
    }
    
    // Process each head in parallel
    parallel_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS UNROLL
        
        // Compute attention scores for this head
        compute_scores: for (int pos = 0; pos < seq_len; pos++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            float score = 0.0f;
            int head_start = head * kHeadDim;
            int k_offset = pos * kDim + head_start;
            
            // Dot product between query head and key head at position pos
            dot_product: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS UNROLL factor=4
                score += q_heads[head][i] * k_cache[k_offset + i];
            }
            attn_scores[pos] = score * scale;
        }
        
        // Apply softmax to attention scores - use separate buffer to avoid in-place operation
        static float attn_scores_softmax[kSeqLen];
#pragma HLS ARRAY_PARTITION variable=attn_scores_softmax cyclic factor=4 dim=1
        
        softmax_kernel(attn_scores, attn_scores_softmax, seq_len);
        
        // Apply attention weights to values
        apply_attention: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            float weighted_sum = 0.0f;
            int head_start = head * kHeadDim;
            
            weighted_sum_loop: for (int pos = 0; pos < seq_len; pos++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
                int v_offset = pos * kDim + head_start + i;
                weighted_sum += attn_scores_softmax[pos] * v_cache[v_offset];
            }
            attn_output_heads[head][i] = weighted_sum;
        }
    }
    
    // Concatenate heads
    concat_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS PIPELINE II=1
        int head_start = head * kHeadDim;
        for (int i = 0; i < kHeadDim; i++) {
#pragma HLS UNROLL factor=4
            output[head_start + i] = attn_output_heads[head][i];
        }
    }
}

// FFN layer implementation
void ffn_layer_kernel(
    float* input,
    float* output,
    float* w1, float* w2, float* w3,
    float* norm_weights
) {
#pragma HLS INLINE OFF
    
    static float normed_input[kDim];
    static float w1_out[kFFNDim];
    static float w3_out[kFFNDim];
    static float silu_out[kFFNDim];
    static float gated_out[kFFNDim];
    
#pragma HLS ARRAY_PARTITION variable=normed_input cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=w1_out cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=w3_out cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=silu_out cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=gated_out cyclic factor=4 dim=1
    
    // 1. Pre-normalization
    rmsnorm_kernel(input, normed_input, norm_weights);
    
    // 2. First linear layer (W1)
    matmul_kernel(normed_input, w1, w1_out, kDim, kFFNDim);
    
    // 3. Gate projection (W3)
    matmul_kernel(normed_input, w3, w3_out, kDim, kFFNDim);
    
    // 4. SiLU activation on W1 output
    silu_activation_kernel(w1_out, silu_out, kFFNDim);
    
    // 5. Element-wise multiplication (gating)
    gating: for (int i = 0; i < kFFNDim; i++) {
#pragma HLS PIPELINE II=1
        gated_out[i] = silu_out[i] * w3_out[i];
    }
    
    // 6. Second linear layer (W2)
    matmul_kernel(gated_out, w2, output, kFFNDim, kDim);
}

// RMSNorm implementation - calls external kernel
void rmsnorm_kernel(
    float* input,
    float* output,
    float* weights
) {
#pragma HLS INLINE OFF
    
    // Call external kernel implementation
    kernel_rmsnorm(input, weights, output, kDim);
}

// RoPE implementation - calls external kernel
void rope_kernel(
    float* q_input,
    float* k_input,
    float* q_output,
    float* k_output,
    float* cos_vals,
    float* sin_vals
) {
#pragma HLS INLINE OFF
    
    // Call external kernel implementation for each head
    rope_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
        int head_start = head * kHeadDim;
        kernel_rope(q_input, k_input, cos_vals, sin_vals, q_output, k_output, head_start);
    }
}

// Matrix multiplication kernel - calls external implementation
void matmul_kernel(
    float* input,
    float* weights,
    float* output,
    int input_size,
    int output_size
) {
#pragma HLS INLINE OFF
    
    // Call external kernel implementation
    kernel_matmul(input, weights, output, input_size, output_size);
}

// SiLU activation kernel - calls external implementation
void silu_activation_kernel(
    float* input,
    float* output,
    int size
) {
#pragma HLS INLINE OFF
    
    // Call external kernel implementation
    kernel_silu(input, output, size);
}

// Softmax kernel - calls external implementation
void softmax_kernel(
    float* input,
    float* output,
    int size
) {
#pragma HLS INLINE OFF
    
    // Call external kernel implementation
    kernel_softmax(input, output, size);
}
} 
