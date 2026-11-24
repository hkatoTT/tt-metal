# Source files for ttnn_op_experimental_deepseek_prefill_decode_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DECODE_MATMUL_API_HEADERS decode_matmul.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DECODE_MATMUL_SRCS
    device/decode_matmul_device_operation.cpp
    device/decode_matmul_program_factory.cpp
    decode_matmul.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/decode_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DECODE_MATMUL_NANOBIND_SRCS decode_matmul_nanobind.cpp)
