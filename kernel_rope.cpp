

#include <hls_stream.h>
#include <cmath>
#include <stdint.h>
#include "tensor.hpp"

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=768
    inStream << i_vec[i];
  }
}

static void compute_rope(hls::stream<float>& q_in_stream,
                         hls::stream<float>& k_in_stream,
                         hls::stream<float>& cos_vec_stream,
                         hls::stream<float>& sin_vec_stream,
                         hls::stream<float>& q_out_stream,
                         hls::stream<float>& k_out_stream, int head_begin) {
#pragma HLS INLINE OFF

  // Use smaller buffers and avoid large arrays
  static float q_local[kDim];
  static float k_local[kDim];
  static float cos_local[kHalvedHeadDim];
  static float sin_local[kHalvedHeadDim];
  static float q_out_local[kDim];
  static float k_out_local[kDim];

  // Conservative array partitioning to avoid resource explosion
#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=k_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=cos_local cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=sin_local cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=q_out_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=k_out_local cyclic factor=4 dim=1

  // Load data with moderate pipelining
  load_qk: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    q_local[i] = q_in_stream.read();
    k_local[i] = k_in_stream.read();
  }

  load_cossin: for (int i = 0; i < kHalvedHeadDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
    cos_local[i] = cos_vec_stream.read();
    sin_local[i] = sin_vec_stream.read();
  }

  // Initialize output arrays
  init_output: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    q_out_local[i] = q_local[i];
    k_out_local[i] = k_local[i];
  }

  // Apply RoPE transformation with limited unrolling
  rope_transform: for (int i = 0; i < kHalvedHeadDim; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
    int i0 = head_begin + i * 2 + 0;
    int i1 = head_begin + i * 2 + 1;

    float q0 = q_local[i0];
    float q1 = q_local[i1];

    float k0 = k_local[i0];
    float k1 = k_local[i1];

    float cos = cos_local[i];
    float sin = sin_local[i];

    q_out_local[i0] = q0 * cos - q1 * sin;
    q_out_local[i1] = q0 * sin + q1 * cos;

    k_out_local[i0] = k0 * cos - k1 * sin;
    k_out_local[i1] = k0 * sin + k1 * cos;
  }

  store_output: for (int i = 0; i < kDim; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    q_out_stream << q_out_local[i];
    k_out_stream << k_out_local[i];
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int vec_size) {
#pragma HLS INLINE OFF
mem_wr:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_rope(float* q_in, float* k_in, float* cos_vec, float* sin_vec,
                 float* q_out, float* k_out, int head_begin) {
#pragma HLS INTERFACE m_axi port = q_in bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = k_in bundle = gmem1 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = cos_vec bundle = gmem2 max_widen_bitwidth = \
    32
#pragma HLS INTERFACE m_axi port = sin_vec bundle = gmem3 max_widen_bitwidth = \
    32
#pragma HLS INTERFACE m_axi port = q_out bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = k_out bundle = gmem1 max_widen_bitwidth = 32
#pragma HLS INTERFACE s_axilite port = head_begin bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static hls::stream<float> q_input_stream("rope_q_input_stream");
  static hls::stream<float> k_input_stream("rope_k_input_stream");
  static hls::stream<float> cos_input_stream("rope_cos_input_stream");
  static hls::stream<float> sin_input_stream("rope_sin_input_stream");
  static hls::stream<float> q_output_stream("rope_q_output_stream");
  static hls::stream<float> k_output_stream("rope_k_output_stream");

#pragma HLS STREAM variable=q_input_stream depth=32
#pragma HLS STREAM variable=k_input_stream depth=32
#pragma HLS STREAM variable=cos_input_stream depth=16
#pragma HLS STREAM variable=sin_input_stream depth=16
#pragma HLS STREAM variable=q_output_stream depth=32
#pragma HLS STREAM variable=k_output_stream depth=32

// Remove dataflow to avoid conflict with pipelined loop in top function
// #pragma HLS dataflow
  load_vec(q_in, q_input_stream, kDim);
  load_vec(k_in, k_input_stream, kDim);
  load_vec(cos_vec, cos_input_stream, kHalvedHeadDim);
  load_vec(sin_vec, sin_input_stream, kHalvedHeadDim);
  compute_rope(q_input_stream, k_input_stream, cos_input_stream, sin_input_stream,
               q_output_stream, k_output_stream, head_begin);
  store_result(q_out, q_output_stream, kDim);
  store_result(k_out, k_output_stream, kDim);
}
}


