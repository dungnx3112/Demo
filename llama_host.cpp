#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>

// Include Vitis HLS headers for host application
#include "xcl2.hpp"
#include "llama_hls_top.hpp"

class LLaMAHost {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    
    // Device memory buffers
    cl::Buffer buffer_input_embedding;
    cl::Buffer buffer_output_logits;
    cl::Buffer buffer_weights[10];  // All weight buffers
    cl::Buffer buffer_k_cache;
    cl::Buffer buffer_v_cache;
    cl::Buffer buffer_cos_table;
    cl::Buffer buffer_sin_table;
    
    // Host memory
    std::vector<float> token_embedding_table;
    std::vector<float> model_weights[10];
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    
public:
    LLaMAHost(const std::string& xclbin_path) {
        // Initialize OpenCL
        auto devices = xcl::get_xil_devices();
        auto device = devices[0];
        
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        
        // Load xclbin
        auto fileBuf = xcl::read_binary_file(xclbin_path);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        program = cl::Program(context, {device}, bins);
        kernel = cl::Kernel(program, "llama_inference_hls_top");
        
        // Initialize memory
        initializeMemory();
    }
    
    void initializeMemory() {
        // Initialize token embedding table [kVocabSize][kDim]
        token_embedding_table.resize(kVocabSize * kDim);
        
        // Initialize model weights (normally loaded from file)
        for (int i = 0; i < 10; i++) {
            model_weights[i].resize(getWeightSize(i));
        }
        
        // Initialize KV cache [kNumLayers][kSeqLen][kDim]
        k_cache.resize(kNumLayers * kSeqLen * kDim, 0.0f);
        v_cache.resize(kNumLayers * kSeqLen * kDim, 0.0f);
        
        // Initialize RoPE tables
        cos_table.resize(kSeqLen * kHeadDim / 2);
        sin_table.resize(kSeqLen * kHeadDim / 2);
        initializeRoPETables();
        
        // Create device buffers
        createDeviceBuffers();
    }
    
private:
    size_t getWeightSize(int weight_idx) {
        switch(weight_idx) {
            case 0: return kNumLayers * kDim * kDim; // attention_wq
            case 1: return kNumLayers * kDim * kDim; // attention_wk
            case 2: return kNumLayers * kDim * kDim; // attention_wv
            case 3: return kNumLayers * kDim * kDim; // attention_wo
            case 4: return kNumLayers * kFFNDim * kDim; // ffn_w1
            case 5: return kNumLayers * kDim * kFFNDim; // ffn_w2
            case 6: return kNumLayers * kFFNDim * kDim; // ffn_w3
            case 7: return kNumLayers * kDim; // attention_norm
            case 8: return kNumLayers * kDim; // ffn_norm
            case 9: return kDim; // final_norm
            default: return 0;
        }
    }
    
    void initializeRoPETables() {
        // Initialize RoPE precomputed sin/cos tables
        const float theta = 10000.0f;
        
        for (int pos = 0; pos < kSeqLen; pos++) {
            for (int i = 0; i < kHeadDim / 2; i++) {
                float angle = pos / powf(theta, 2.0f * i / kHeadDim);
                cos_table[pos * (kHeadDim / 2) + i] = cosf(angle);
                sin_table[pos * (kHeadDim / 2) + i] = sinf(angle);
            }
        }
    }
    
    void createDeviceBuffers() {
        buffer_input_embedding = cl::Buffer(context, CL_MEM_READ_ONLY, kDim * sizeof(float));
        buffer_output_logits = cl::Buffer(context, CL_MEM_WRITE_ONLY, kVocabSize * sizeof(float));
        
        // Weight buffers
        for (int i = 0; i < 10; i++) {
            buffer_weights[i] = cl::Buffer(context, CL_MEM_READ_ONLY, 
                                         getWeightSize(i) * sizeof(float));
        }
        
        // KV cache buffers
        buffer_k_cache = cl::Buffer(context, CL_MEM_READ_WRITE, 
                                   kNumLayers * kSeqLen * kDim * sizeof(float));
        buffer_v_cache = cl::Buffer(context, CL_MEM_READ_WRITE, 
                                   kNumLayers * kSeqLen * kDim * sizeof(float));
        
        // RoPE tables
        buffer_cos_table = cl::Buffer(context, CL_MEM_READ_ONLY, 
                                     kSeqLen * kHeadDim / 2 * sizeof(float));
        buffer_sin_table = cl::Buffer(context, CL_MEM_READ_ONLY, 
                                     kSeqLen * kHeadDim / 2 * sizeof(float));
    }
    
public:
    // Token ID to embedding conversion (CPU-side)
    std::vector<float> tokenToEmbedding(int token_id) {
        std::vector<float> embedding(kDim);
        if (token_id >= 0 && token_id < kVocabSize) {
            std::copy(
                token_embedding_table.begin() + token_id * kDim,
                token_embedding_table.begin() + (token_id + 1) * kDim,
                embedding.begin()
            );
        }
        return embedding;
    }
    
    // Single token inference
    std::vector<float> inferenceStep(const std::vector<float>& input_embedding, 
                                   int position) {
        std::vector<float> output_logits(kVocabSize);
        
        // Transfer input to device
        queue.enqueueWriteBuffer(buffer_input_embedding, CL_TRUE, 0, 
                               kDim * sizeof(float), input_embedding.data());
        
        // Set kernel arguments
        int arg_idx = 0;
        kernel.setArg(arg_idx++, buffer_input_embedding);
        kernel.setArg(arg_idx++, buffer_output_logits);
        
        // Set weight buffers
        for (int i = 0; i < 10; i++) {
            kernel.setArg(arg_idx++, buffer_weights[i]);
        }
        
        // Set KV cache buffers
        kernel.setArg(arg_idx++, buffer_k_cache);
        kernel.setArg(arg_idx++, buffer_v_cache);
        
        // Set RoPE tables
        kernel.setArg(arg_idx++, buffer_cos_table);
        kernel.setArg(arg_idx++, buffer_sin_table);
        
        // Set control parameters
        kernel.setArg(arg_idx++, position);
        kernel.setArg(arg_idx++, kSeqLen);
        
        // Execute kernel
        queue.enqueueTask(kernel);
        queue.finish();
        
        // Read back results
        queue.enqueueReadBuffer(buffer_output_logits, CL_TRUE, 0, 
                              kVocabSize * sizeof(float), output_logits.data());
        
        return output_logits;
    }
    
    // Generate sequence of tokens
    std::vector<int> generateSequence(const std::vector<int>& prompt_tokens, 
                                    int max_new_tokens, 
                                    float temperature = 1.0f) {
        std::vector<int> generated_tokens = prompt_tokens;
        
        // Clear KV cache at start of generation
        clearKVCache();
        
        std::cout << "Starting sequence generation..." << std::endl;
        std::cout << "Prompt length: " << prompt_tokens.size() << std::endl;
        
        // Process prompt tokens
        for (int i = 0; i < prompt_tokens.size(); i++) {
            auto embedding = tokenToEmbedding(prompt_tokens[i]);
            auto logits = inferenceStep(embedding, i);
            std::cout << "Processed prompt token " << i << "/" << prompt_tokens.size() << std::endl;
        }
        
        // Generate new tokens
        for (int i = 0; i < max_new_tokens; i++) {
            int current_pos = prompt_tokens.size() + i;
            int last_token = generated_tokens.back();
            
            // Convert last token to embedding
            auto embedding = tokenToEmbedding(last_token);
            
            // Run inference
            auto start_time = std::chrono::high_resolution_clock::now();
            auto logits = inferenceStep(embedding, current_pos);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            // Sample next token (simple greedy sampling for now)
            int next_token = sampleToken(logits, temperature);
            generated_tokens.push_back(next_token);
            
            std::cout << "Generated token " << (i + 1) << "/" << max_new_tokens 
                     << " (ID: " << next_token << ") in " << duration << "ms" << std::endl;
            
            // Check for end of sequence
            if (next_token == 2) { // EOS token
                std::cout << "End of sequence detected." << std::endl;
                break;
            }
        }
        
        return generated_tokens;
    }
    
private:
    void clearKVCache() {
        // Clear host cache
        std::fill(k_cache.begin(), k_cache.end(), 0.0f);
        std::fill(v_cache.begin(), v_cache.end(), 0.0f);
        
        // Clear device cache
        queue.enqueueWriteBuffer(buffer_k_cache, CL_TRUE, 0, 
                               k_cache.size() * sizeof(float), k_cache.data());
        queue.enqueueWriteBuffer(buffer_v_cache, CL_TRUE, 0, 
                               v_cache.size() * sizeof(float), v_cache.data());
    }
    
    int sampleToken(const std::vector<float>& logits, float temperature) {
        // Simple greedy sampling (choose highest probability)
        int best_token = 0;
        float best_score = logits[0];
        
        for (int i = 1; i < logits.size(); i++) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_token = i;
            }
        }
        
        return best_token;
    }
    
public:
    // Load model weights from file
    bool loadModelWeights(const std::string& weights_path) {
        std::ifstream file(weights_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open weights file: " << weights_path << std::endl;
            return false;
        }
        
        // Load token embedding table
        file.read(reinterpret_cast<char*>(token_embedding_table.data()), 
                 token_embedding_table.size() * sizeof(float));
        
        // Load model weights
        for (int i = 0; i < 10; i++) {
            file.read(reinterpret_cast<char*>(model_weights[i].data()), 
                     model_weights[i].size() * sizeof(float));
        }
        
        file.close();
        
        // Transfer weights to device
        transferWeightsToDevice();
        
        std::cout << "Model weights loaded successfully." << std::endl;
        return true;
    }
    
private:
    void transferWeightsToDevice() {
        // Transfer all weights to device
        for (int i = 0; i < 10; i++) {
            queue.enqueueWriteBuffer(buffer_weights[i], CL_TRUE, 0, 
                                   model_weights[i].size() * sizeof(float), 
                                   model_weights[i].data());
        }
        
        // Transfer RoPE tables
        queue.enqueueWriteBuffer(buffer_cos_table, CL_TRUE, 0, 
                               cos_table.size() * sizeof(float), cos_table.data());
        queue.enqueueWriteBuffer(buffer_sin_table, CL_TRUE, 0, 
                               sin_table.size() * sizeof(float), sin_table.data());
    }
};

// Example usage
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin_path> [weights_path]" << std::endl;
        return 1;
    }
    
    std::string xclbin_path = argv[1];
    std::string weights_path = (argc > 2) ? argv[2] : "llama_weights.bin";
    
    try {
        // Initialize LLaMA host
        LLaMAHost llama(xclbin_path);
        
        // Load model weights
        if (!llama.loadModelWeights(weights_path)) {
            std::cerr << "Failed to load model weights." << std::endl;
            return 1;
        }
        
        // Example prompt
        std::vector<int> prompt = {1, 15043, 1827, 338};  // Example token IDs
        
        // Generate sequence
        auto generated = llama.generateSequence(prompt, 50);
        
        // Print results
        std::cout << "\nGenerated sequence:" << std::endl;
        for (int token : generated) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}