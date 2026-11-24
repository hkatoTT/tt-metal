// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "writer_decode_prepare.hpp"

void kernel_main() { kda_prepare_writer::run<0, 0>(); }
