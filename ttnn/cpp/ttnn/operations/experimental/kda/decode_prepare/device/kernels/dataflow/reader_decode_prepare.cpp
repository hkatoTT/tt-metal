// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reader_decode_prepare.hpp"

void kernel_main() { kda_prepare_reader::run<0, 0>(); }
