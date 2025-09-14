# LLaMA HLS Modular Architecture

## 📁 Project Structure (Refactored)

### Core Files:
- **`llama_hls_top.cpp/.hpp`**: Main pipeline orchestration only
- **`tensor.hpp`**: Constants và type definitions  
- **`tensor_fpga.cpp/.hpp`**: FPGA utility functions

### Compute Kernel Files (Separate for debugging):
- **`kernel_matmul.cpp`**: Matrix multiplication implementation
- **`kernel_rmsnorm.cpp`**: RMS normalization implementation  
- **`kernel_rope.cpp`**: Rotary Position Embedding implementation
- **`kernel_softmax.cpp`**: Softmax attention weights implementation
- **`kernel_silu.cpp`**: SiLU activation function implementation

### Support Files:
- **`testbench_hls.cpp`**: HLS-compatible testing
- **`llama_host.cpp`**: Host application for OpenCL
- **`llama_synthesis.tcl`**: Synthesis script

## 🔧 Architecture Benefits

### 1. **Modular Design**
- ✅ Each compute operation isolated in separate file
- ✅ Easy to debug individual kernels
- ✅ Independent testing và optimization
- ✅ Clear separation of concerns

### 2. **Development Workflow**
```bash
# Test individual kernels
vitis_hls -f test_kernel_matmul.tcl
vitis_hls -f test_kernel_rmsnorm.tcl
# etc.

# Test integrated pipeline
vitis_hls -f llama_synthesis.tcl
```

### 3. **Debug Strategy**
1. **Individual Kernel Testing**: Test mỗi kernel riêng lẻ
2. **Integration Testing**: Test pipeline với all kernels
3. **Performance Optimization**: Optimize từng kernel independently

## 🎯 Function Mapping

### llama_hls_top.cpp Functions:
```cpp
// Wrapper functions - just call external kernels
void matmul_kernel() -> kernel_matmul()
void rmsnorm_kernel() -> kernel_rmsnorm() 
void rope_kernel() -> kernel_rope()
void softmax_kernel() -> kernel_softmax()
void silu_activation_kernel() -> kernel_silu()

// High-level orchestration functions
void attention_layer_with_cache_kernel()
void ffn_layer_kernel()
void llama_inference_hls_top() // Main top function
```

### External Kernel Files:
```cpp
// kernel_matmul.cpp
extern "C" void kernel_matmul(float* i_vec, float* i_mat, float* o_vec, int vec_size, int col_size)

// kernel_rmsnorm.cpp  
extern "C" void kernel_rmsnorm(float* i_vec_1, float* i_vec_2, float* o_vec, int vec_size)

// kernel_rope.cpp
extern "C" void kernel_rope(float* q_in, float* k_in, float* cos_vec, float* sin_vec,
                            float* q_out, float* k_out, int head_begin)

// kernel_softmax.cpp
extern "C" void kernel_softmax(float* i_vec, float* o_vec, int vec_size)

// kernel_silu.cpp
extern "C" void kernel_silu(float* i_vec, float* o_vec, int vec_size)
```

## 🚀 Synthesis Flow

### All Kernels Included:
```tcl
# llama_synthesis.tcl includes all files
add_files llama_hls_top.cpp
add_files kernel_matmul.cpp
add_files kernel_rmsnorm.cpp  
add_files kernel_rope.cpp
add_files kernel_softmax.cpp
add_files kernel_silu.cpp
```

### Benefits for Debugging:
1. **Isolated Testing**: Test each kernel with specific inputs
2. **Performance Profiling**: Profile individual kernels
3. **Resource Analysis**: Understand resource usage per kernel
4. **Incremental Development**: Develop và test one kernel at a time

## 📊 Expected Workflow

### Development Phase:
```bash
# 1. Test individual kernels first
# 2. Verify each kernel functionality 
# 3. Integrate into main pipeline
# 4. Full system testing
```

### Debug Phase:
```bash
# If issue occurs:
# 1. Identify problematic kernel
# 2. Test kernel in isolation
# 3. Fix kernel implementation
# 4. Re-integrate and test
```

This modular architecture makes the LLaMA HLS project much more maintainable and debuggable! 🎯