# Source files for ttnn_op_experimental_deepseek_prefill_moe_decode_route.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_DECODE_ROUTE_API_HEADERS moe_decode_route.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_DECODE_ROUTE_SRCS
    device/moe_decode_route_device_operation.cpp
    device/moe_decode_route_program_factory.cpp
    moe_decode_route.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_decode_route/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_DECODE_ROUTE_NANOBIND_SRCS moe_decode_route_nanobind.cpp)
