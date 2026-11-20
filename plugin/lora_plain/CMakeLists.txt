# plugin/lora_plain -- plain bf16 views of rotated int8 weight planes, so the core runtime LoRA
# machinery can bind adapters to W8A8-rotated trunk modules: the site gate accepts a plain fp GEMM
# over a transform-basis weight (core/lora/lora.cpp, op_is_site) and this
# library presents the planes in the plain basis at load so the adapter epilogue sees plain
# activations.
#
# A host-only kernel library: it registers the plain fp `gemm_nt` (dtype bf16) with a layout hook
# that accepts the affine+fwht128 int8 encoding and a relayout that decodes it to the plain basis
# at load (H^2 = 128 I, one rounding to bf16), then delegates the launch to libr4d's own exported
# bf16 GEMM entries via dlopen(RTLD_NOLOAD)/dlsym -- so the arithmetic is byte-for-byte libr4d's,
# only the operand bytes differ. No device sources; it links libdl alone. Built in tree or
# standalone against an installed radiance (docs/PLUGIN.md section 13).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(lora_plain LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(lora_plain kernels
               SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lora_plain.cpp)
target_link_libraries(lora_plain PRIVATE ${CMAKE_DL_LIBS})
