#include <hls_stream.h>
#include <stdint.h>

#define MAX_DATA_SIZE 1024

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    inStream << i_vec[i];
  }
}

static void load_mat(float* i_mat, hls::stream<float>& inStream, int vec_size,
                     int col_size) {
#pragma HLS INLINE OFF
mem_rd:
  for (int i = 0; i < col_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=2048
    for (int j = 0; j < vec_size; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
      inStream << i_mat[vec_size * i + j];
    }
  }
}

static void compute_matmul(hls::stream<float>& in1_stream,
                           hls::stream<float>& in2_stream,
                           hls::stream<float>& out_stream, int vec_size,
                           int col_size) {
#pragma HLS INLINE OFF

  static float vec_local[MAX_DATA_SIZE];
#pragma HLS ARRAY_PARTITION variable=vec_local cyclic factor=4 dim=1

  // Load vector once
  load_vector: for (int i = 0; i < vec_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
    vec_local[i] = in1_stream.read();
  }

execute:
  for (int i = 0; i < col_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=2048
    float sum_local = 0;
    dot_product: for (int j = 0; j < vec_size; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
      sum_local += vec_local[j] * in2_stream.read();
    }
    out_stream << sum_local;
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int col_size) {
#pragma HLS INLINE OFF
mem_wr:
  for (int i = 0; i < col_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=768 max=2048
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_matmul(float* i_vec, float* i_mat, float* o_vec, int vec_size,
                   int col_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = i_mat bundle = gmem1 max_widen_bitwidth = 32
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0 max_widen_bitwidth = 32
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = col_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  // Use unique stream names to avoid conflicts
  static hls::stream<float> vector_stream("matmul_vector_stream");
  static hls::stream<float> matrix_stream("matmul_matrix_stream");
  static hls::stream<float> result_stream("matmul_result_stream");

#pragma HLS STREAM variable=vector_stream depth=64
#pragma HLS STREAM variable=matrix_stream depth=64
#pragma HLS STREAM variable=result_stream depth=64

#pragma HLS dataflow
  load_vec(i_vec, vector_stream, vec_size);
  load_mat(i_mat, matrix_stream, vec_size, col_size);
  compute_matmul(vector_stream, matrix_stream, result_stream, vec_size, col_size);
  store_result(o_vec, result_stream, col_size);
}
}


