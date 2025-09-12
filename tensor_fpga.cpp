#include "tensor_fpga.hpp"

#ifdef __SYNTHESIS__
#include "hls_math.h"
#else
#include <cmath>
#define hls_sqrt sqrtf
#define hls_exp expf
#endif

/* ---------------------------------  
      Basic Arithmetic Operations
   --------------------------------- */

// Add a scalar to each element of the input tensor
void AddFPGA(float* out, const float* in, float a, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    loop_add_scalar: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        out[i] = in[i] + a;
    }
}

// Multiply each element of the input tensor by a scalar
void MulFPGA(float* out, const float* in, float a, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    loop_mul_scalar: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        out[i] = in[i] * a;
    }
}

// Element-wise addition of two tensors
void AddFPGA(float* out, const float* lhs, const float* rhs, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=lhs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=rhs offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    loop_add_element: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        out[i] = lhs[i] + rhs[i];
    }
}

// Element-wise multiplication of two tensors
void MulFPGA(float* out, const float* lhs, const float* rhs, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=lhs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=rhs offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    loop_mul_element: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=3072
        out[i] = lhs[i] * rhs[i];
    }
}

/* ---------------------------------  
        Matrix Multiplication
   --------------------------------- */

// General matrix multiplication
void MatmulFPGA(float* out, const float* in, const float* w, 
                int in_size, int out_size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=in_size bundle=control
#pragma HLS INTERFACE s_axilite port=out_size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for better memory access patterns
    float local_in[768];
    float local_w[768];
    
#pragma HLS ARRAY_PARTITION variable=local_in cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=local_w cyclic factor=16

    // Load input vector
    load_input: for (int i = 0; i < in_size; i++) {
#pragma HLS PIPELINE II=1
        local_in[i] = in[i];
    }

    // Matrix multiplication
    matmul_outer: for (int i = 0; i < out_size; i++) {
        float sum = 0;
        
        // Load weight row
        load_weights: for (int k = 0; k < in_size; k++) {
#pragma HLS PIPELINE II=1
            local_w[k] = w[i * in_size + k];
        }
        
        // Compute dot product
        matmul_inner: for (int k = 0; k < in_size; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
            sum += local_in[k] * local_w[k];
        }
        
        out[i] = sum;
    }
}

// Specialized matrix multiplication for attention
void MatmulAttnFPGA(float* out, const float* in, const float* w, int dim) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=dim bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    MatmulFPGA(out, in, w, dim, dim);
}

// Specialized matrix multiplication for FFN
void MatmulFFNFPGA(float* out, const float* in, const float* w,
                   int in_dim, int out_dim) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=in_dim bundle=control
#pragma HLS INTERFACE s_axilite port=out_dim bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    MatmulFPGA(out, in, w, in_dim, out_dim);
}

/* ---------------------------------  
          Normalization
   --------------------------------- */

// RMS Normalization
void RMSNormFPGA(float* out, const float* in, const float* w, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const float eps = 1e-5f;
    float sum_sq = 0;
    
    // Calculate sum of squares
    rms_sum: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        sum_sq += in[i] * in[i];
    }
    
    // Calculate RMS normalization factor
#ifdef __SYNTHESIS__
    float rms = hls::sqrt(sum_sq / size + eps);
#else
    float rms = hls_sqrt(sum_sq / size + eps);
#endif
    
    // Apply normalization and weight
    rms_norm: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
        out[i] = (in[i] / rms) * w[i];
    }
}

/* ---------------------------------  
        Attention Operations
   --------------------------------- */

// Softmax function
void SoftmaxFPGA(float* out, const float* in, int size, int max_pos) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=max_pos bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float max_val = -1000000.0f;
    int actual_size = (max_pos > 0 && max_pos < size) ? max_pos : size;
    
    // Find maximum value
    find_max: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=1024
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }
    
    float sum_exp = 0;
    
    // Calculate exp and sum
    calc_exp: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=1024
#ifdef __SYNTHESIS__
        float exp_val = hls::exp(in[i] - max_val);
#else
        float exp_val = hls_exp(in[i] - max_val);
#endif
        out[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    normalize: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=1024
        out[i] = out[i] / sum_exp;
    }
    
    // Set remaining values to 0
    zero_rest: for (int i = actual_size; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = 0.0f;
    }
}

/* ---------------------------------  
      Rotary Position Embedding
   --------------------------------- */

// Rotary Position Embedding
void RoPEFPGA(float* q_out, float* k_out, const float* q_in,
              const float* k_in, const float* cos_vec,
              const float* sin_vec, int head_begin, int head_size) {
#pragma HLS INTERFACE m_axi port=q_out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=k_out offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=q_in offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=k_in offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=cos_vec offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=sin_vec offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=head_begin bundle=control
#pragma HLS INTERFACE s_axilite port=head_size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    rope_loop: for (int i = 0; i < head_size / 2; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
        int idx1 = head_begin + i;
        int idx2 = head_begin + i + head_size / 2;
        
        float cos_val = cos_vec[i];
        float sin_val = sin_vec[i];
        
        // Apply RoPE to query
        float q1 = q_in[idx1];
        float q2 = q_in[idx2];
        q_out[idx1] = q1 * cos_val - q2 * sin_val;
        q_out[idx2] = q1 * sin_val + q2 * cos_val;
        
        // Apply RoPE to key
        float k1 = k_in[idx1];
        float k2 = k_in[idx2];
        k_out[idx1] = k1 * cos_val - k2 * sin_val;
        k_out[idx2] = k1 * sin_val + k2 * cos_val;
    }
}

/* ---------------------------------  
        Activation Functions
   --------------------------------- */

// SiLU (Swish) activation function
void SiLUFPGA(float* out, const float* in, int size) {
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    silu_loop: for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=3072 max=3072
        float x = in[i];
#ifdef __SYNTHESIS__
        float sigmoid = 1.0f / (1.0f + hls::exp(-x));
#else
        float sigmoid = 1.0f / (1.0f + hls_exp(-x));
#endif
        out[i] = x * sigmoid;
    }
}

/* ---------------------------------  
          Cache Operations
   --------------------------------- */

// Update KV cache
void UpdateKVCacheFPGA(float* cache, const float* new_kv, 
                       int pos, int head_dim, int layer) {
#pragma HLS INTERFACE m_axi port=cache offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=new_kv offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=pos bundle=control
#pragma HLS INTERFACE s_axilite port=head_dim bundle=control
#pragma HLS INTERFACE s_axilite port=layer bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int cache_offset = layer * 1024 * head_dim + pos * head_dim; // kSeqLen = 1024
    
    update_cache: for (int i = 0; i < head_dim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        cache[cache_offset + i] = new_kv[i];
    }
}

/* ---------------------------------  
        Vocabulary Projection
   --------------------------------- */

// Vocabulary projection (final linear layer)
void VocabProjectionFPGA(float* logits, const float* hidden, 
                         const float* embed_weights, int vocab_size, int dim) {
#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=hidden offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=embed_weights offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=vocab_size bundle=control
#pragma HLS INTERFACE s_axilite port=dim bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffer for input
    float local_hidden[768];
#pragma HLS ARRAY_PARTITION variable=local_hidden cyclic factor=16

    // Load hidden state
    load_hidden: for (int i = 0; i < dim; i++) {
#pragma HLS PIPELINE II=1
        local_hidden[i] = hidden[i];
    }

    // Compute logits for each vocabulary token
    vocab_outer: for (int i = 0; i < vocab_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=32000 max=32000
        float sum = 0;
        
        vocab_inner: for (int j = 0; j < dim; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
            sum += local_hidden[j] * embed_weights[i * dim + j];
        }
        
        logits[i] = sum;
    }
}
