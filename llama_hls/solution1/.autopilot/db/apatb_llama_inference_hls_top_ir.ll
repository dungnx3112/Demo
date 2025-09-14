; ModuleID = 'C:/Academic_Sem5/DA1/Demo/llama_hls/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

; Function Attrs: noinline willreturn
define void @apatb_llama_inference_hls_top_ir(float* noalias nocapture nonnull readonly "maxi" %input_embedding, float* noalias nocapture nonnull "maxi" %output_logits, float* noalias nocapture nonnull readonly "maxi" %weight_token_embedding, float* noalias nocapture nonnull readonly "maxi" %weight_attention_wq, float* noalias nocapture nonnull readonly "maxi" %weight_attention_wk, float* noalias nocapture nonnull readonly "maxi" %weight_attention_wv, float* noalias nocapture nonnull readonly "maxi" %weight_attention_wo, float* noalias nocapture nonnull readonly "maxi" %weight_ffn_w1, float* noalias nocapture nonnull readonly "maxi" %weight_ffn_w2, float* noalias nocapture nonnull readonly "maxi" %weight_ffn_w3, float* noalias nocapture nonnull readonly "maxi" %weight_attention_norm, float* noalias nocapture nonnull readonly "maxi" %weight_ffn_norm, float* noalias nocapture nonnull readonly "maxi" %weight_final_norm, float* noalias nocapture nonnull "maxi" %k_cache, float* noalias nocapture nonnull "maxi" %v_cache, float* noalias nocapture nonnull readonly "maxi" %cos_table, float* noalias nocapture nonnull readonly "maxi" %sin_table, i32 %position, i32 %max_position) local_unnamed_addr #0 {
entry:
  %input_embedding_copy = alloca float, align 512
  %output_logits_copy = alloca float, align 512
  %weight_token_embedding_copy = alloca float, align 512
  %weight_attention_wq_copy = alloca float, align 512
  %weight_attention_wk_copy = alloca float, align 512
  %weight_attention_wv_copy = alloca float, align 512
  %weight_attention_wo_copy = alloca float, align 512
  %weight_ffn_w1_copy = alloca float, align 512
  %weight_ffn_w2_copy = alloca float, align 512
  %weight_ffn_w3_copy = alloca float, align 512
  %weight_attention_norm_copy = alloca float, align 512
  %weight_ffn_norm_copy = alloca float, align 512
  %weight_final_norm_copy = alloca float, align 512
  %k_cache_copy = alloca float, align 512
  %v_cache_copy = alloca float, align 512
  %cos_table_copy = alloca float, align 512
  %sin_table_copy = alloca float, align 512
  call fastcc void @copy_in(float* nonnull %input_embedding, float* nonnull align 512 %input_embedding_copy, float* nonnull %output_logits, float* nonnull align 512 %output_logits_copy, float* nonnull %weight_token_embedding, float* nonnull align 512 %weight_token_embedding_copy, float* nonnull %weight_attention_wq, float* nonnull align 512 %weight_attention_wq_copy, float* nonnull %weight_attention_wk, float* nonnull align 512 %weight_attention_wk_copy, float* nonnull %weight_attention_wv, float* nonnull align 512 %weight_attention_wv_copy, float* nonnull %weight_attention_wo, float* nonnull align 512 %weight_attention_wo_copy, float* nonnull %weight_ffn_w1, float* nonnull align 512 %weight_ffn_w1_copy, float* nonnull %weight_ffn_w2, float* nonnull align 512 %weight_ffn_w2_copy, float* nonnull %weight_ffn_w3, float* nonnull align 512 %weight_ffn_w3_copy, float* nonnull %weight_attention_norm, float* nonnull align 512 %weight_attention_norm_copy, float* nonnull %weight_ffn_norm, float* nonnull align 512 %weight_ffn_norm_copy, float* nonnull %weight_final_norm, float* nonnull align 512 %weight_final_norm_copy, float* nonnull %k_cache, float* nonnull align 512 %k_cache_copy, float* nonnull %v_cache, float* nonnull align 512 %v_cache_copy, float* nonnull %cos_table, float* nonnull align 512 %cos_table_copy, float* nonnull %sin_table, float* nonnull align 512 %sin_table_copy)
  call void @apatb_llama_inference_hls_top_hw(float* %input_embedding_copy, float* %output_logits_copy, float* %weight_token_embedding_copy, float* %weight_attention_wq_copy, float* %weight_attention_wk_copy, float* %weight_attention_wv_copy, float* %weight_attention_wo_copy, float* %weight_ffn_w1_copy, float* %weight_ffn_w2_copy, float* %weight_ffn_w3_copy, float* %weight_attention_norm_copy, float* %weight_ffn_norm_copy, float* %weight_final_norm_copy, float* %k_cache_copy, float* %v_cache_copy, float* %cos_table_copy, float* %sin_table_copy, i32 %position, i32 %max_position)
  call void @copy_back(float* %input_embedding, float* %input_embedding_copy, float* %output_logits, float* %output_logits_copy, float* %weight_token_embedding, float* %weight_token_embedding_copy, float* %weight_attention_wq, float* %weight_attention_wq_copy, float* %weight_attention_wk, float* %weight_attention_wk_copy, float* %weight_attention_wv, float* %weight_attention_wv_copy, float* %weight_attention_wo, float* %weight_attention_wo_copy, float* %weight_ffn_w1, float* %weight_ffn_w1_copy, float* %weight_ffn_w2, float* %weight_ffn_w2_copy, float* %weight_ffn_w3, float* %weight_ffn_w3_copy, float* %weight_attention_norm, float* %weight_attention_norm_copy, float* %weight_ffn_norm, float* %weight_ffn_norm_copy, float* %weight_final_norm, float* %weight_final_norm_copy, float* %k_cache, float* %k_cache_copy, float* %v_cache, float* %v_cache_copy, float* %cos_table, float* %cos_table_copy, float* %sin_table, float* %sin_table_copy)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_in(float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512) unnamed_addr #1 {
entry:
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %1, float* %0)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %3, float* %2)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %5, float* %4)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %7, float* %6)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %9, float* %8)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %11, float* %10)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %13, float* %12)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %15, float* %14)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %17, float* %16)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %19, float* %18)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %21, float* %20)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %23, float* %22)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %25, float* %24)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %27, float* %26)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %29, float* %28)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %31, float* %30)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %33, float* %32)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0f32(float* noalias align 512 %dst, float* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq float* %dst, null
  %1 = icmp eq float* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %3 = load float, float* %src, align 4
  store float %3, float* %dst, align 512
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_out(float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512) unnamed_addr #3 {
entry:
  call fastcc void @onebyonecpy_hls.p0f32(float* %0, float* align 512 %1)
  call fastcc void @onebyonecpy_hls.p0f32(float* %2, float* align 512 %3)
  call fastcc void @onebyonecpy_hls.p0f32(float* %4, float* align 512 %5)
  call fastcc void @onebyonecpy_hls.p0f32(float* %6, float* align 512 %7)
  call fastcc void @onebyonecpy_hls.p0f32(float* %8, float* align 512 %9)
  call fastcc void @onebyonecpy_hls.p0f32(float* %10, float* align 512 %11)
  call fastcc void @onebyonecpy_hls.p0f32(float* %12, float* align 512 %13)
  call fastcc void @onebyonecpy_hls.p0f32(float* %14, float* align 512 %15)
  call fastcc void @onebyonecpy_hls.p0f32(float* %16, float* align 512 %17)
  call fastcc void @onebyonecpy_hls.p0f32(float* %18, float* align 512 %19)
  call fastcc void @onebyonecpy_hls.p0f32(float* %20, float* align 512 %21)
  call fastcc void @onebyonecpy_hls.p0f32(float* %22, float* align 512 %23)
  call fastcc void @onebyonecpy_hls.p0f32(float* %24, float* align 512 %25)
  call fastcc void @onebyonecpy_hls.p0f32(float* %26, float* align 512 %27)
  call fastcc void @onebyonecpy_hls.p0f32(float* %28, float* align 512 %29)
  call fastcc void @onebyonecpy_hls.p0f32(float* %30, float* align 512 %31)
  call fastcc void @onebyonecpy_hls.p0f32(float* %32, float* align 512 %33)
  ret void
}

declare void @apatb_llama_inference_hls_top_hw(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, i32, i32)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_back(float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512) unnamed_addr #3 {
entry:
  call fastcc void @onebyonecpy_hls.p0f32(float* %2, float* align 512 %3)
  call fastcc void @onebyonecpy_hls.p0f32(float* %26, float* align 512 %27)
  call fastcc void @onebyonecpy_hls.p0f32(float* %28, float* align 512 %29)
  ret void
}

declare void @llama_inference_hls_top_hw_stub(float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull, float* noalias nocapture nonnull, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, i32, i32)

define void @llama_inference_hls_top_hw_stub_wrapper(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, i32, i32) #4 {
entry:
  call void @copy_out(float* null, float* %0, float* null, float* %1, float* null, float* %2, float* null, float* %3, float* null, float* %4, float* null, float* %5, float* null, float* %6, float* null, float* %7, float* null, float* %8, float* null, float* %9, float* null, float* %10, float* null, float* %11, float* null, float* %12, float* null, float* %13, float* null, float* %14, float* null, float* %15, float* null, float* %16)
  call void @llama_inference_hls_top_hw_stub(float* %0, float* %1, float* %2, float* %3, float* %4, float* %5, float* %6, float* %7, float* %8, float* %9, float* %10, float* %11, float* %12, float* %13, float* %14, float* %15, float* %16, i32 %17, i32 %18)
  call void @copy_in(float* null, float* %0, float* null, float* %1, float* null, float* %2, float* null, float* %3, float* null, float* %4, float* null, float* %5, float* null, float* %6, float* null, float* %7, float* null, float* %8, float* null, float* %9, float* null, float* %10, float* null, float* %11, float* null, float* %12, float* null, float* %13, float* null, float* %14, float* null, float* %15, float* null, float* %16)
  ret void
}

attributes #0 = { noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #4 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
