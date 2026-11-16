set(TTNN_OP_EXPERIMENTAL_KDA_DECODE_PREPARE_API_HEADERS decode_prepare.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_DECODE_PREPARE_SRCS
    decode_prepare.cpp
    device/decode_prepare_device_operation.cpp
    device/decode_prepare_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_DECODE_PREPARE_NANOBIND_SRCS decode_prepare_nanobind.cpp)
