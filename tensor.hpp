#ifndef TENSOR_HPP_
#define TENSOR_HPP_

// Vitis HLS headers - only include when synthesizing
#ifdef __SYNTHESIS__
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"
#else
// Fallback headers for simulation/testbench
#include <cmath>
#include <cstdint>
#include <string>
#endif

// LLaMA2 100M parameters configuration
constexpr int kDim = 768;              // Hidden dimension (768 for 100M model)
constexpr int kVocabSize = 32000;      // Vocabulary size
constexpr int kNumLayers = 12;         // Number of transformer layers (12 for 100M)
constexpr int kNumHeads = 12;          // Number of attention heads (12 for 100M)
constexpr int kNumKVHeads = 12;        // Number of key-value heads (same as heads for 100M)
constexpr int kHeadDim = kDim / kNumHeads;  // Dimension per head (64)
constexpr int kSinCosTable = kHeadDim; // RoPE sin/cos table size
constexpr int kSeqLen = 1024;          // Maximum sequence length
constexpr int kFFNDim = 3072;          // FFN intermediate dimension (4 * kDim)
constexpr int kHalvedHeadDim = kHeadDim / 2;  // Half head dimension for RoPE

using Tensor1d = float[kDim];
using Tensor2dTok = float[kVocabSize][kDim];
using Tensor2dAttn = float[kDim][kDim];
using Tensor3dAttn = float[kNumLayers][kDim][kDim];
using Tensor2dRMS = float[kNumLayers][kDim];
using Tensor1dSinCos = float[kSinCosTable];
using Tensor2dSinCos = float[kSeqLen][kSinCosTable];
using Tensor2dFFNA = float[kFFNDim][kDim];
using Tensor3dFFNA = float[kNumLayers][kFFNDim][kDim];
using Tensor1dFFNB = float[kFFNDim];
using Tensor2dFFNB = float[kDim][kFFNDim];
using Tensor3dFFNB = float[kNumLayers][kDim][kFFNDim];
using Tensor1dQKSM = float[kSeqLen];
using Tensor2dQKSM = float[kNumLayers][kSeqLen];
using Tensor2dFFNC = float[kNumLayers][kFFNDim];
using Tensor2dCache = float[kSeqLen][kDim];
using Tensor3dCache = float[kNumLayers][kSeqLen][kDim];
using Tensor1dLogits = float[kVocabSize];
using Tensor2d = float[kDim][kDim];
using Tensor3d = float[kDim][kDim][kDim];

// FPGA optimized tensor operations using Vitis HLS
void CopyTensor1d(Tensor1d& dst, const Tensor1d& src);
void CopyTensor2d(Tensor2d& dst, const Tensor2d& src);
void CopyTensor3d(Tensor3d& dst, const Tensor3d& src);

// FPGA implementations of tensor operations
void Add(Tensor1d& out, const Tensor1d& in, float a);
void Mul(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a);
void Sub(Tensor1d& out, const Tensor1d& in, float a);
void Div(Tensor1d& out, const Tensor1d& in, float a);

void Add(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);
void Mul(Tensor1dFFNB& out, const Tensor1dFFNB& lhs, const Tensor1dFFNB& rhs);
void Sub(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);
void Div(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);

float InnerProduct(const Tensor1d& lhs, const Tensor1d& rhs);
void Matmul(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w);
void MutmulRanged(Tensor1dQKSM& out, const Tensor1d& in, const Tensor2dCache& w,
                  int i_begin, int i_end, int j_begin, int j_end);
void MutmulRangedTranspose(Tensor1d& out, const Tensor1dQKSM& in,
                           const Tensor2dCache& w, int i_begin, int i_end,
                           int j_begin, int j_end);
void Matmul(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w);
void Matmul(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w);
void MutmulVocab(Tensor1dLogits& out, const Tensor1d& in, const Tensor2dTok& w);

// Additional overloads for compatibility
void Mul(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a);
void Matmul(Tensor1dFFNB& out, const Tensor1d& in, const Tensor3dFFNA& w, int layer);
void Matmul(Tensor1d& out, const Tensor1dFFNB& in, const Tensor3dFFNB& w, int layer);

void ReLU(Tensor1d& out, const Tensor1d& in);
void SiLU(Tensor1dFFNB& out, const Tensor1dFFNB& in);

void RMSNorm(Tensor1d& out, const Tensor1d& in, const Tensor1d& w);
void Softmax(Tensor1dQKSM& out, const Tensor1dQKSM& in, int max_pos = -1);
void Softmax(Tensor1dLogits& out, const Tensor1dLogits& in, int max_pos = -1);

#ifndef __SYNTHESIS__
std::pair<int, float> FindMaxIndexAndValue(const Tensor1d& in);
#endif
float Max(const Tensor1d& in);
int Argmax(const Tensor1dLogits& values);

void RoPE(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
          const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
          const Tensor1dSinCos& sin_vec, int head_begin, int head_size);

#endif // TENSOR_HPP_
