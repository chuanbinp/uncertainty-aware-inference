# """
# Quantization configurations for Llama-2 13B PTQ sweep.

# Each config defines the loading method, precision, and HuggingFace model ID.
# Update model IDs here if using different quantized checkpoints.
# """

# QUANT_CONFIGS = {
#     "gptq_int8": {
#         "method": "gptq",
#         "bits": 8,
#         "precision": "int8",
#         "quant_method": "gptq",
#         "hf_model_id": "TheBloke/Llama-2-13B-GPTQ",
#         "gptq_revision": "gptq-8bit-128g-actorder_True",
#     },
#     "gptq_int4": {
#         "method": "gptq",
#         "bits": 4,
#         "precision": "int4",
#         "quant_method": "gptq",
#         "hf_model_id": "TheBloke/Llama-2-13B-GPTQ",
#         "gptq_revision": "gptq-4bit-128g-actorder_True",
#     },
#     "awq_int4": {
#         "method": "awq",
#         "bits": 4,
#         "precision": "int4",
#         "quant_method": "awq",
#         "hf_model_id": "TheBloke/Llama-2-13B-AWQ",
#     },
#     "bnb_nf4": {
#         "method": "bitsandbytes",
#         "bits": 4,
#         "precision": "nf4",
#         "quant_method": "bitsandbytes",
#         "hf_model_id": "meta-llama/Llama-2-13b-hf",
#     },
# }