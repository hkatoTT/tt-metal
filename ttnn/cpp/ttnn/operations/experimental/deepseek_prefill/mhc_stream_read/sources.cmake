# Source files for ttnn_op_experimental_deepseek_prefill_mhc_stream_read.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_READ_API_HEADERS mhc_stream_read.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_READ_SRCS
    device/mhc_stream_read_device_operation.cpp
    device/mhc_stream_read_program_factory.cpp
    mhc_stream_read.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_read/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_STREAM_READ_NANOBIND_SRCS mhc_stream_read_nanobind.cpp)
