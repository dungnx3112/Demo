#include <hls_stream.h>
#include <cmath>
#include <stdint.h>
#include "tensor.hpp"

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    inStream << i_vec[i];
  }
}

static void compute_silu(hls::stream<float>& in_stream,
                         hls::stream<float>& out_stream, int vec_size) {
#pragma HLS INLINE OFF

  static float vec_local[kFFNDim];
#pragma HLS ARRAY_PARTITION variable=vec_local cyclic factor=4 dim=1
  
  load_data: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    vec_local[i] = in_stream.read();
  }

  compute_activation: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    float x = vec_local[i];
    float sigmoid = 1.0f / (1.0f + expf(-x));
    vec_local[i] = x * sigmoid;
  }

  store_data: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    out_stream << vec_local[i];
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int vec_size) {
#pragma HLS INLINE OFF
mem_wr:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_silu(float* i_vec, float* o_vec, int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static hls::stream<float> input_stream("silu_input_stream");
  static hls::stream<float> output_stream("silu_output_stream");

#pragma HLS STREAM variable=input_stream depth=32
#pragma HLS STREAM variable=output_stream depth=32

#pragma HLS dataflow
  load_vec(i_vec, input_stream, vec_size);
  compute_silu(input_stream, output_stream, vec_size);
  store_result(o_vec, output_stream, vec_size);
}
}