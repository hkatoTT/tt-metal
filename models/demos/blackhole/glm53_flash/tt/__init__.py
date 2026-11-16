"""CPU (torch) reference of the GLM-5.3-Flash text model, one module per component.

Every component follows the Hugging Face Transformers `glm5_next` code (5.20.2) operation by operation, including where
it casts between bf16 and fp32, so it can be compared with HF bit for bit.  Weights are views of the checkpoint
(`checkpoint.Glm53Checkpoint`), so a layer costs no memory until its pages are read.  Every device test compares
against these modules.
"""
