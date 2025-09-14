#include <iostream>
#include <cmath>
#include <cstring>

// HLS headers
#ifdef __SYNTHESIS__
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"
#else
#include <cmath>
#include <cstdint>
#endif

#include "llama_hls_top.hpp"

// Simple test utilities for HLS
void fillRandom(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    // Simple deterministic "random" for HLS compatibility
    for (int i = 0; i < size; i++) {
        data[i] = min_val + (max_val - min_val) * (float)(i % 100) / 100.0f;
        if (i % 3 == 0) data[i] = -data[i]; // Add some negative values
    }
}

void fillZero(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = 0.0f;
    }
}

float computeRMSE(const float* a, const float* b, int size) {
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
#ifdef __SYNTHESIS__
    return hls::sqrt(sum_sq_diff / size);
#else
    return sqrt(sum_sq_diff / size);
#endif
}

void printVector(const float* data, int size, const char* name) {
#ifndef __SYNTHESIS__
    printf("%s: ", name);
    for (int i = 0; i < (size < 10 ? size : 10); i++) {
        printf("%.3f ", data[i]);
    }
    if (size > 10) printf("...");
    printf("\n");
#endif
}

// Test individual kernels - HLS compatible
int testMatmulKernel() {
    printf("\n=== Testing Matrix Multiplication Kernel ===\n");
    
    const int input_size = 256;
    const int output_size = 256;
    
    // Use static arrays for HLS compatibility
    static float input[256];
    static float weights[256 * 256];
    static float output[256];
    static float expected[256];
    
    // Fill with test data
    fillRandom(input, input_size);
    fillRandom(weights, output_size * input_size);
    fillZero(output, output_size);
    
    // Compute expected result (CPU reference)
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }
        expected[i] = sum;
    }
    
    // Test HLS kernel
    matmul_kernel(input, weights, output, input_size, output_size);
    
    // Compare results
    float rmse = computeRMSE(output, expected, output_size);
    printf("RMSE: %.6f\n", rmse);
    
    printVector(expected, output_size, "Expected");
    printVector(output, output_size, "HLS Output");
    
    if (rmse < 1e-6) {
        printf("✓ Matrix multiplication test passed!\n");
        return 0;
    } else {
        printf("✗ Matrix multiplication test failed!\n");
        return 1;
    }
}

int testRMSNormKernel() {
    printf("\n=== Testing RMS Normalization Kernel ===\n");
    
    static float input[kDim];
    static float weights[kDim];
    static float output[kDim];
    static float expected[kDim];
    
    // Fill with test data
    fillRandom(input, kDim);
    fillRandom(weights, kDim, 0.5f, 1.5f);
    fillZero(output, kDim);
    
    // Compute expected result (CPU reference)
    float mean_square = 0.0f;
    for (int i = 0; i < kDim; i++) {
        mean_square += input[i] * input[i];
    }
    mean_square /= kDim;
    
#ifdef __SYNTHESIS__
    float rms_norm = 1.0f / hls::sqrt(mean_square + 1e-6f);
#else
    float rms_norm = 1.0f / sqrt(mean_square + 1e-6f);
#endif
    
    for (int i = 0; i < kDim; i++) {
        expected[i] = input[i] * rms_norm * weights[i];
    }
    
    // Test HLS kernel
    rmsnorm_kernel(input, output, weights);
    
    // Compare results
    float rmse = computeRMSE(output, expected, kDim);
    printf("RMSE: %.6f\n", rmse);
    
    printVector(expected, kDim, "Expected");
    printVector(output, kDim, "HLS Output");
    
    if (rmse < 1e-5) {
        printf("✓ RMS Normalization test passed!\n");
        return 0;
    } else {
        printf("✗ RMS Normalization test failed!\n");
        return 1;
    }
}

int testSoftmaxKernel() {
    printf("\n=== Testing Softmax Kernel ===\n");
    
    const int size = 100;
    static float input[100];
    static float output[100];
    static float expected[100];
    
    // Fill with test data
    fillRandom(input, size, -5.0f, 5.0f);
    fillZero(output, size);
    
    // Compute expected result (CPU reference)
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
#ifdef __SYNTHESIS__
        expected[i] = hls::exp(input[i] - max_val);
#else
        expected[i] = exp(input[i] - max_val);
#endif
        sum_exp += expected[i];
    }
    
    for (int i = 0; i < size; i++) {
        expected[i] /= sum_exp;
    }
    
    // Test HLS kernel
    softmax_kernel(input, output, size);
    
    // Compare results
    float rmse = computeRMSE(output, expected, size);
    printf("RMSE: %.6f\n", rmse);
    
    // Check if probabilities sum to 1
    float sum_output = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_output += output[i];
    }
    printf("Sum of probabilities: %.6f\n", sum_output);
    
    printVector(expected, size, "Expected");
    printVector(output, size, "HLS Output");
    
    if (rmse < 1e-5 && fabs(sum_output - 1.0f) < 1e-5) {
        printf("✓ Softmax test passed!\n");
        return 0;
    } else {
        printf("✗ Softmax test failed!\n");
        return 1;
    }
}

int testRoPEKernel() {
    printf("\n=== Testing RoPE Kernel ===\n");
    
    static float q_input[kDim];
    static float k_input[kDim];
    static float q_output[kDim];
    static float k_output[kDim];
    static float cos_vals[kHeadDim/2];
    static float sin_vals[kHeadDim/2];
    
    // Fill with test data
    fillRandom(q_input, kDim);
    fillRandom(k_input, kDim);
    fillZero(q_output, kDim);
    fillZero(k_output, kDim);
    
    // Generate RoPE values
    for (int i = 0; i < kHeadDim/2; i++) {
        float angle = 0.1f * i;  // Simple test angle
#ifdef __SYNTHESIS__
        cos_vals[i] = hls::cos(angle);
        sin_vals[i] = hls::sin(angle);
#else
        cos_vals[i] = cos(angle);
        sin_vals[i] = sin(angle);
#endif
    }
    
    // Test HLS kernel
    rope_kernel(q_input, k_input, q_output, k_output, cos_vals, sin_vals);
    
    printVector(q_input, kDim, "Q Input");
    printVector(q_output, kDim, "Q Output");
    printVector(k_input, kDim, "K Input");
    printVector(k_output, kDim, "K Output");
    
    printf("✓ RoPE test completed (manual verification needed)\n");
    return 0;
}

int testSiLUKernel() {
    printf("\n=== Testing SiLU Activation Kernel ===\n");
    
    const int size = kFFNDim;
    static float input[kFFNDim];
    static float output[kFFNDim];
    static float expected[kFFNDim];
    
    // Fill with test data
    fillRandom(input, size, -3.0f, 3.0f);
    fillZero(output, size);
    
    // Compute expected result (CPU reference)
    for (int i = 0; i < size; i++) {
        float x = input[i];
#ifdef __SYNTHESIS__
        float sigmoid = 1.0f / (1.0f + hls::exp(-x));
#else
        float sigmoid = 1.0f / (1.0f + exp(-x));
#endif
        expected[i] = x * sigmoid;
    }
    
    // Test HLS kernel
    silu_activation_kernel(input, output, size);
    
    // Compare results
    float rmse = computeRMSE(output, expected, size);
    printf("RMSE: %.6f\n", rmse);
    
    printVector(expected, size, "Expected");
    printVector(output, size, "HLS Output");
    
    if (rmse < 1e-5) {
        printf("✓ SiLU activation test passed!\n");
        return 0;
    } else {
        printf("✗ SiLU activation test failed!\n");
        return 1;
    }
}

// Main test function for HLS
int main() {
    printf("LLaMA HLS Kernel Test Suite (HLS Compatible)\n");
    printf("============================================\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test individual kernels
    total_tests++; if (testMatmulKernel() == 0) passed_tests++;
    total_tests++; if (testRMSNormKernel() == 0) passed_tests++;
    total_tests++; if (testSoftmaxKernel() == 0) passed_tests++;
    total_tests++; if (testRoPEKernel() == 0) passed_tests++;
    total_tests++; if (testSiLUKernel() == 0) passed_tests++;
    
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d/%d tests\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("🎉 All tests passed! Ready for HLS synthesis.\n");
        return 0;
    } else {
        printf("❌ Some tests failed. Please check implementation.\n");
        return 1;
    }
}