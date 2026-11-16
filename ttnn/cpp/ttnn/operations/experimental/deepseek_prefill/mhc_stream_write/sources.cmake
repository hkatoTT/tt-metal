# Source files for ttnn_op_experimental_deepseek_prefill_mhc_stream_write.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_WRITE_API_HEADERS mhc_stream_write.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_WRITE_SRCS
    device/mhc_stream_write_device_operation.cpp
    device/mhc_stream_write_program_factory.cpp
    mhc_stream_write.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_write/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_WRITE_NANOBIND_SRCS mhc_stream_write_nanobind.cpp)
