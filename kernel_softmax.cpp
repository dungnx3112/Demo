

#include <hls_stream.h>
#include <cmath>
#include <stdint.h>
#include "tensor.hpp"

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=1024
    inStream << i_vec[i];
  }
}

static void compute_softmax(hls::stream<float>& in_stream,
                            hls::stream<float>& out_stream, int vec_size) {
#pragma HLS INLINE OFF

  static float vec_local_1[kSeqLen];  // Use kSeqLen as max buffer size  
  static float vec_local_2[kSeqLen];

#pragma HLS ARRAY_PARTITION variable=vec_local_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=vec_local_2 cyclic factor=4 dim=1

  int actual_size = (vec_size == -1) ? kSeqLen : vec_size;
  
  // Load input data
  load_data: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=1024
    vec_local_1[i] = in_stream.read();
  }

  // 1. Find Max
  float max_val = vec_local_1[0];
  find_max: for (int i = 1; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=31 max=1023
    if (vec_local_1[i] > max_val) {
      max_val = vec_local_1[i];
    }
  }

  // 2. Compute exp and sum
  float sum = 0;
  compute_exp: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=1024
    vec_local_2[i] = expf(vec_local_1[i] - max_val);
    sum += vec_local_2[i];
  }

  // 3. Normalize and write output
  normalize: for (int i = 0; i < actual_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=1024
    float normalized = vec_local_2[i] / sum;
    out_stream << normalized;
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int vec_size) {
#pragma HLS INLINE OFF
mem_wr:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=32 max=1024
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_softmax(float* i_vec, float* o_vec, int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  // Use unique stream names to avoid conflicts
  static hls::stream<float> input_stream("softmax_input_stream");
  static hls::stream<float> output_stream("softmax_output_stream");

#pragma HLS STREAM variable=input_stream depth=64
#pragma HLS STREAM variable=output_stream depth=64

  // Ensure positive size
  int actual_size = (vec_size <= 0) ? kSeqLen : vec_size;
  if (actual_size > kSeqLen) actual_size = kSeqLen;

#pragma HLS dataflow
  load_vec(i_vec, input_stream, actual_size);
  compute_softmax(input_stream, output_stream, actual_size);
  store_result(o_vec, output_stream, actual_size);
}
}


