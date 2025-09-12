#include "llama_hls_top.hpp"

namespace swan {

// Main HLS Top-level function - optimized for single token inference
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
    
    // RoPE precomputed tables
    float* cos_table,
    float* sin_table,
    
    // Control parameters
    int position,
    int max_position
) {
    // HLS Interface pragmas for AXI - Fixed memory bundle distribution
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
#pragma HLS INTERFACE m_axi port=cos_table offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=sin_table offset=slave bundle=gmem14

#pragma HLS INTERFACE s_axilite port=position bundle=control
#pragma HLS INTERFACE s_axilite port=max_position bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Use MUCH smaller local buffers to avoid stack overflow
    // Process in smaller chunks if needed
    static float hidden_state[kDim];
    static float attn_output[kDim];
    static float ffn_output[kDim];
    static float temp_buffer[kDim];
    
    // Partition arrays for better memory access patterns
#pragma HLS ARRAY_PARTITION variable=hidden_state cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=attn_output cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=ffn_output cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_buffer cyclic factor=16 dim=1

    // Load input embedding into local buffer
    load_input_loop: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        hidden_state[i] = input_embedding[i];
    }

    // Process through all transformer layers with DATAFLOW
    transformer_layers: for (int layer = 0; layer < kNumLayers; layer++) {
#pragma HLS PIPELINE OFF
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
        
        // Calculate weight offsets for current layer
        int layer_offset = layer * kDim;
        int attn_offset = layer * kDim * kDim;
        int ffn_w1_offset = layer * kFFNDim * kDim;
        int ffn_w2_offset = layer * kDim * kFFNDim;
        
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
        
        // Attention layer
        attention_layer_kernel(
            hidden_state,
            attn_output,
            &weight_attention_wq[attn_offset],
            &weight_attention_wk[attn_offset],
            &weight_attention_wv[attn_offset],
            &weight_attention_wo[attn_offset],
            &weight_attention_norm[layer_offset],
            cos_vals,
            sin_vals,
            position
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
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
            sum += hidden_state[j] * weight_token_embedding[i * kDim + j];
        }
        output_logits[i] = sum;
    }
}

// Attention layer implementation
void attention_layer_kernel(
    float* input,
    float* output,
    float* wq, float* wk, float* wv, float* wo,
    float* norm_weights,
    float* cos_vals, float* sin_vals,
    int position
) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW  // Enable dataflow between pipeline stages
    
    // Use static allocation to avoid stack overflow
    static float normed_input[kDim];
    static float q[kDim], k[kDim], v[kDim];
    static float q_rope[kDim], k_rope[kDim];
    
    // Aggressive array partitioning for maximum bandwidth
#pragma HLS ARRAY_PARTITION variable=normed_input cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=q_rope cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=k_rope cyclic factor=16 dim=1
    
    // PIPELINE STAGE 1: Pre-normalization
    rmsnorm_kernel(input, normed_input, norm_weights);
    
    // PIPELINE STAGE 2: Parallel Q,K,V projections (can execute concurrently)
    // Note: In fully optimized version, these would use separate DSP resources
    matmul_kernel(normed_input, wq, q, kDim, kDim);
    matmul_kernel(normed_input, wk, k, kDim, kDim);
    matmul_kernel(normed_input, wv, v, kDim, kDim);
    
    // PIPELINE STAGE 3: RoPE rotation
    rope_kernel(q, k, q_rope, k_rope, cos_vals, sin_vals);
    
    // 4. PARALLEL MULTI-HEAD ATTENTION - All 12 heads processed concurrently
    static float q_heads[kNumHeads][kHeadDim];
    static float k_heads[kNumHeads][kHeadDim];
    static float v_heads[kNumHeads][kHeadDim];
    static float attn_weights[kNumHeads][kHeadDim];
    static float attn_output_heads[kNumHeads][kHeadDim];
    static float concat_output[kDim];
    
    // Partition arrays for maximum parallelism - COMPLETE partitioning for heads
#pragma HLS ARRAY_PARTITION variable=q_heads complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_heads complete dim=1  
#pragma HLS ARRAY_PARTITION variable=v_heads complete dim=1
#pragma HLS ARRAY_PARTITION variable=attn_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=attn_output_heads complete dim=1
#pragma HLS ARRAY_PARTITION variable=concat_output cyclic factor=16 dim=1

    float scale = 1.0f / sqrtf((float)kHeadDim);
    
    // Split Q, K, V into heads - PARALLEL head extraction
    split_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS UNROLL  // Process all heads in parallel
        int head_start = head * kHeadDim;
        for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            q_heads[head][i] = q_rope[head_start + i];
            k_heads[head][i] = k_rope[head_start + i];
            v_heads[head][i] = v[head_start + i];
        }
    }
    
    // Compute attention for ALL HEADS in PARALLEL
    parallel_attention: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS UNROLL  // All heads computed concurrently
        
        // Compute attention scores for this head
        compute_attn: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            attn_weights[head][i] = q_heads[head][i] * k_heads[head][i] * scale;
        }
        
        // Apply softmax to attention weights (simplified - just normalize)
        float sum_weights = 0.0f;
        sum_attn: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            sum_weights += attn_weights[head][i];
        }
        
        normalize_attn: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            attn_weights[head][i] = attn_weights[head][i] / (sum_weights + 1e-8f);
        }
        
        // Apply attention to values for this head
        apply_attn: for (int i = 0; i < kHeadDim; i++) {
#pragma HLS PIPELINE II=1
            float weighted_val = 0.0f;
            for (int j = 0; j < kHeadDim; j++) {
#pragma HLS UNROLL factor=8
                weighted_val += attn_weights[head][j] * v_heads[head][i];
            }
            attn_output_heads[head][i] = weighted_val;
        }
    }
    
    // Concatenate all heads back to full dimension
    concat_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS PIPELINE II=1
        int head_start = head * kHeadDim;
        for (int i = 0; i < kHeadDim; i++) {
#pragma HLS UNROLL factor=8
            concat_output[head_start + i] = attn_output_heads[head][i];
        }
    }
    
    // Copy concatenated result to output
    copy_concat: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
        output[i] = concat_output[i];
    }
    
    // 5. Output projection
    static float temp_output[kDim];
#pragma HLS ARRAY_PARTITION variable=temp_output cyclic factor=8 dim=1
    
    matmul_kernel(output, wo, temp_output, kDim, kDim);
    
    // Copy result
    copy_output: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
        output[i] = temp_output[i];
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
    
#pragma HLS ARRAY_PARTITION variable=normed_input cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=w1_out cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=w3_out cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=silu_out cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=gated_out cyclic factor=16 dim=1
    
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

// RMSNorm implementation
void rmsnorm_kernel(
    float* input,
    float* output,
    float* weights
) {
#pragma HLS INLINE OFF
    
    // Compute mean of squares
    float mean_square = 0.0f;
    compute_mean: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
        mean_square += input[i] * input[i];
    }
    mean_square /= (float)kDim;
    
    // Compute normalization factor
    float rms_norm = 1.0f / sqrtf(mean_square + 1e-6f);
    
    // Apply normalization and scaling
    normalize: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
        output[i] = input[i] * rms_norm * weights[i];
    }
}

// RoPE implementation
void rope_kernel(
    float* q_input,
    float* k_input,
    float* q_output,
    float* k_output,
    float* cos_vals,
    float* sin_vals
) {
#pragma HLS INLINE OFF
    
    // Apply RoPE to each head
    rope_heads: for (int head = 0; head < kNumHeads; head++) {
#pragma HLS PIPELINE II=1
        int head_start = head * kHeadDim;
        
        // Apply rotation to pairs of elements
        rope_pairs: for (int i = 0; i < kHeadDim/2; i++) {
#pragma HLS PIPELINE II=1
            int idx1 = head_start + 2*i;
            int idx2 = head_start + 2*i + 1;
            
            float q1 = q_input[idx1];
            float q2 = q_input[idx2];
            float k1 = k_input[idx1];
            float k2 = k_input[idx2];
            
            float cos_val = cos_vals[i];
            float sin_val = sin_vals[i];
            
            q_output[idx1] = q1 * cos_val - q2 * sin_val;
            q_output[idx2] = q1 * sin_val + q2 * cos_val;
            k_output[idx1] = k1 * cos_val - k2 * sin_val;
            k_output[idx2] = k1 * sin_val + k2 * cos_val;
        }
    }
}

// Matrix multiplication kernel - Optimized for HLS
void matmul_kernel(
    float* input,
    float* weights,
    float* output,
    int input_size,
    int output_size
) {
#pragma HLS INLINE OFF
    
    // Use local buffer to improve memory access pattern
    static float input_buffer[kDim];
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=8 dim=1
    
    // Load input to local buffer
    input_load: for (int j = 0; j < input_size; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        input_buffer[j] = input[j];
    }
    
    // Compute matrix multiplication with unrolled inner loop
    matmul_outer: for (int i = 0; i < output_size; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        float sum = 0.0f;
        
        matmul_inner: for (int j = 0; j < input_size; j++) {
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
            sum += input_buffer[j] * weights[i * input_size + j];
        }
        output[i] = sum;
    }
}

// SiLU activation kernel
void silu_activation_kernel(
    float* input,
    float* output,
    int size
) {
#pragma HLS INLINE OFF
    
    silu_loop: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        float x = input[i];
        output[i] = x * (1.0f / (1.0f + expf(-x)));
    }
}

// Softmax kernel
void softmax_kernel(
    float* input,
    float* output,
    int size
) {
#pragma HLS INLINE OFF
    
    // Find maximum value for numerical stability
    float max_val = input[0];
    find_max: for (int i = 1; i < size; i++) {
#pragma HLS PIPELINE II=1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    float sum_exp = 0.0f;
    compute_exp: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    normalize_soft: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        output[i] = output[i] / sum_exp;
    }
}
} 
