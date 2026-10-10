# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""EXPERIMENT (hkato): ethernet dispatch on the WH Galaxy 4x8 (8x10 compute grid instead of 8x9).

Every ethernet channel of a WH Galaxy is cabled (torus) and used by the fabric, so there is no idle ethernet core
for dispatch. This script lists the ethernet channels of K links per SP-axis edge (both ends of every link) for
HK_DISABLE_ETH_CHANS; tt_cluster then treats those links as down, the fabric skips them, and their ethernet cores
become idle cores for prefetch / dispatch / dispatch_s. K=2 gives each chip 4 idle cores (the standard 1-CQ
ethernet dispatch needs 3) and leaves the SP axis 2 links.

Usage (after a tt-smi -glx_reset, with the stock dispatch): python gen_disable_list.py [K] > disable_sp_k2.txt
"""
import sys

import tt_umd

import ttnn


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    desc = tt_umd.TopologyDiscovery.discover(tt_umd.TopologyDiscoveryOptions())[0]
    conn = desc.get_ethernet_connections()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    ids = list(mesh.get_device_ids())  # row-major mesh order; the SP axis (axis 1) runs along a row
    ttnn.close_mesh_device(mesh)
    out = []
    for r in range(4):
        row = ids[r * 8 : (r + 1) * 8]
        for i in range(8):
            a, b = row[i], row[(i + 1) % 8]
            chans = sorted(ch for ch, (dst, _) in conn[a].items() if dst == b)
            for ch in chans[-k:]:
                dst, dst_ch = conn[a][ch]
                out += [f"{a}:{ch}", f"{dst}:{dst_ch}"]
    print(",".join(out))


if __name__ == "__main__":
    main()
