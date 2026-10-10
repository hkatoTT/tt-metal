# EXPERIMENT (hkato): ethernet dispatch on the WH Galaxy 4x8 (8x10 compute grid). Source before a Wan run.
# Two of the four links of every SP-axis edge leave the fabric; their ethernet cores run prefetch, dispatch and
# dispatch_s (the stock 1-CQ ethernet dispatch). The SP axis keeps 2 links (WAN_AXIS_NUM_LINKS=1:2).
# The taken links stay down after the run (the dispatch firmware replaces the ethernet base firmware) until
# tt-smi -glx_reset; the 2-channel mesh graph descriptor lets the next runs map the mesh either way.
# HK_DISABLE_ETH_CHANS is machine specific: regenerate it with gen_disable_list.py on another Galaxy.
D=$(dirname "${BASH_SOURCE[0]}")
export HK_ETH_DISPATCH=1
export HK_DISABLE_ETH_CHANS=$(cat ${HK_ETH_DISABLE_LIST:-$D/disable_sp_k2_uf_ev_b5_gwh01.txt})
export TT_MESH_GRAPH_DESC_PATH=$(realpath $D/galaxy_torus_xy_ch2.textproto)
export WAN_AXIS_NUM_LINKS=1:2
# the matmuls keep their 8x9 blockings; QKV and FFN1 use the extra row
export WAN_MM_GRID=8,9 WAN_AGMM_WIDE_N=3840,3456
export "WAN_AGMM_OVR_89=8192x5120x3840=16,8,14,4,2;8192x5120x3456=16,8,12,4,2"
