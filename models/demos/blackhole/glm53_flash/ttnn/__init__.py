"""Device (TT-NN) side of GLM-5.3-Flash on the BH Galaxy 8x4 mesh, one module per component like `tt/`.

Placement: users split over the 8 rows (mesh axis 0); attention, the dense MLP and the shared expert tensor-parallel
over the 4 chips of a row (mesh axis 1); the routed experts sliced by mesh column (column c's 72 experts, chip (r, c)
their intermediate columns 256 r .. 256 r + 255: `moe.py`).
"""
