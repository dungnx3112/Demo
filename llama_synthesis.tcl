open_project llama_hls -reset
set_top llama_inference_hls_top

# Add source files (only existing ones)
add_files llama_hls_top.cpp
add_files llama_hls_top.hpp
add_files tensor.hpp
add_files tensor_fpga.cpp
add_files tensor_fpga.hpp

add_files kernel_matmul.cpp
add_files kernel_rmsnorm.cpp
add_files kernel_rope.cpp
add_files kernel_softmax.cpp
add_files kernel_silu.cpp


# Create solution with realistic optimization
open_solution "solution1" -reset -flow_target vivado
set_part xcv80-lsva4737-2MHP-e-S
create_clock -period 4.0 -name default


puts "Starting C synthesis for LLaMA inference..."
csynth_design

puts "Synthesis completed successfully!"
exit