import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Bypasses TorchInductor ASTSource mismatch

from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
# Configure the model with TFLA Triton kernels
xlstm_config = xLSTMLargeConfig(
    embedding_dim=64,
    num_heads=2,
    num_blocks=6,
    vocab_size=32,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--triton_xl_chunk",
    sequence_kernel="native_sequence__triton",
    step_kernel="triton",
)

# Instantiate the model
xlstm = xLSTMLarge(xlstm_config).to("cuda")

# 1. Print PyTorch Architecture & Parameter Summary
print("=== MODEL ARCHITECTURE ===")
print(xlstm)

total_params = sum(p.numel() for p in xlstm.parameters())
trainable_params = sum(p.numel() for p in xlstm.parameters() if p.requires_grad)
print(f"\nTotal Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Estimated FP32 Size:  {total_params * 4 / (1024**2):.2f} MB\n")

# # 2. Create inputs and print input shape
# input_ids = torch.randint(0, 2048, (3, 256)).to("cuda")
# print("=== TENSOR SHAPES ===")
# print(f"Input shape:          {input_ids.shape}  # (batch_size, sequence_length)")

# # 3. Run forward pass and print output shapes
# out = xlstm(input_ids)

# # Handle tuple output when return_last_states=True
# if isinstance(out, tuple):
#     logits, last_states = out
#     print(f"Output logits shape:  {logits.shape}  # (batch_size, sequence_length, vocab_size)")
#     print(f"Number of block states returned: {len(last_states)}")
# else:
#     print(f"Output shape:         {out.shape}")