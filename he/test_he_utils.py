import tenseal as ts
import torch
import numpy as np
from he_utils import create_context, encrypt_tensor, decrypt_chunks_to_tensor

print("TenSEAL version:", ts.__version__)

# Create a smaller context just for testing
ctx = create_context(poly_mod_degree=8192)

# random vector
x = torch.randn(1000)

# encrypt
enc_chunks, shape = encrypt_tensor(ctx, x)

# add all chunks (homomorphic)
enc_sum = enc_chunks[0]
for c in enc_chunks[1:]:
    enc_sum = enc_sum + c

# decrypt
dec = decrypt_chunks_to_tensor(ctx, [enc_sum], shape)

print("Decoded length:", len(dec.flatten()))
print("First 5 values:", dec.flatten()[:5])
