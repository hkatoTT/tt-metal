# EXPERIMENT (hkato): ethernet dispatch on the WH Galaxy 4x8 (8x10 compute grid). Source before a Wan run.
# One of the four links of every SP-axis edge leaves the fabric; its ethernet cores (2 per chip) run prefetch and
# dispatch (1-CQ ethernet dispatch without dispatch_s, HK_ETH_NO_DISPATCH_S). The SP axis keeps 3 links; the ring
# SDPA all-gather then needs the bank-owned packet schedule (TT_RING_AG_BANK_OWNED_WH): the stock Wormhole path
# splits each head into contiguous per-link page ranges, and with 3 links (odd ranges) it is not deterministic.
# HK_ETH_LINKS_TAKEN=2 instead takes 2 links per edge (stock 1-CQ ethernet dispatch with dispatch_s, 2 SP links).
# The taken links stay down after the run (the dispatch firmware replaces the ethernet base firmware) until
# tt-smi -glx_reset; the mesh graph descriptor with the matching channel count maps the mesh either way. Going from
# 2 taken links to 1, or back to the stock dispatch, needs a reset. The first run after a reset can fail (links
# still training / first takeover of the ethernet cores); run once more.
# HK_DISABLE_ETH_CHANS is machine specific: regenerate it with gen_disable_list.py on another Galaxy.
D=$(dirname "${BASH_SOURCE[0]}")
K=${HK_ETH_LINKS_TAKEN:-1}
export HK_ETH_DISPATCH=1
export HK_DISABLE_ETH_CHANS=$(cat ${HK_ETH_DISABLE_LIST:-$D/disable_sp_k${K}_uf_ev_b5_gwh01.txt})
export TT_MESH_GRAPH_DESC_PATH=$(realpath $D/galaxy_torus_xy_ch$((4 - K)).textproto)
export WAN_AXIS_NUM_LINKS=1:$((4 - K))
if [ "$K" = 1 ]; then
  export HK_ETH_NO_DISPATCH_S=1 TT_RING_AG_BANK_OWNED_WH=1
fi
# the matmuls keep their 8x9 blockings; QKV and FFN1 use the extra row
export WAN_MM_GRID=8,9 WAN_AGMM_WIDE_N=3840,3456
export "WAN_AGMM_OVR_89=8192x5120x3840=16,8,14,4,2;8192x5120x3456=16,8,12,4,2"
