

#include <hls_stream.h>
#include <cmath>
#include <stdint.h>
#include "tensor.hpp"

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    inStream << i_vec[i];
  }
}

static void compute_rmsnorm(hls::stream<float>& in1_stream,
                            hls::stream<float>& in2_stream,
                            hls::stream<float>& out_stream, int vec_size) {
#pragma HLS INLINE OFF

  static float vec_local_1[kDim];
  static float vec_local_2[kDim];
  
#pragma HLS ARRAY_PARTITION variable=vec_local_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=vec_local_2 cyclic factor=4 dim=1

  float sum_local = 0;
  
  load_input: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    vec_local_1[i] = in1_stream.read();
  }
  
  load_weights: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    vec_local_2[i] = in2_stream.read();
  }

  compute_sum: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    sum_local += vec_local_1[i] * vec_local_1[i];
  }

  constexpr float eps = 1e-6f;
  const float norm = 1.0f / sqrtf(sum_local / (float)vec_size + eps);

  normalize: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    out_stream << vec_local_1[i] * norm * vec_local_2[i];
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
void kernel_rmsnorm(float* i_vec_1, float* i_vec_2, float* o_vec,
                    int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec_1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = i_vec_2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static hls::stream<float> input1_stream("rmsnorm_input1_stream");
  static hls::stream<float> input2_stream("rmsnorm_input2_stream");
  static hls::stream<float> output_stream("rmsnorm_output_stream");

#pragma HLS STREAM variable=input1_stream depth=32
#pragma HLS STREAM variable=input2_stream depth=32
#pragma HLS STREAM variable=output_stream depth=32

#pragma HLS dataflow
  load_vec(i_vec_1, input1_stream, vec_size);
  load_vec(i_vec_2, input2_stream, vec_size);
  compute_rmsnorm(input1_stream, input2_stream, output_stream, vec_size);
  store_result(o_vec, output_stream, vec_size);
}
}


