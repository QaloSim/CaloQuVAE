import torch

def decode_binary_energy(x_encoded, lin_bits=23):
    """
    Decodes the original energy value from the linear bits portion 
    of the encoded tensor.
    
    Args:
        x_encoded (torch.Tensor): The full encoded tensor, shape (batch_size, n_latent_nodes).
        lin_bits (int): The number of bits used for the linear encoding (e.g., 23).
        
    Returns:
        torch.Tensor: The decoded energy values, shape (batch_size, 1).
    """
    
    # --- 1. Extract the linear encoding ---
    # We only take the first `lin_bits` columns, which represent the
    # direct binary encoding.
    linear_encoding = x_encoded[:, :lin_bits]
    
    # --- 2. Create the bit-weight mask ---
    # This is the inverse of the mask in the `binary` function.
    # It's a tensor like [1, 2, 4, 8, ..., 2**(lin_bits-1)]
    mask = 2**torch.arange(lin_bits).to(x_encoded.device, x_encoded.dtype)
    
    # --- 3. Multiply and Sum ---
    # We multiply the binary encoding (0s and 1s) by their corresponding
    # bit weights and sum them up.
    # (batch_size, lin_bits) * (lin_bits,) -> (batch_size, lin_bits)
    weighted_bits = linear_encoding * mask
    
    # Sum along dimension 1 to get the final integer value for each item in the batch
    # (batch_size, lin_bits) -> (batch_size,)
    decoded_values = torch.sum(weighted_bits, dim=1)
    
    # --- 4. Reshape and Return ---
    # Add a dimension to match the original input shape (batch_size, 1)
    return decoded_values.unsqueeze(1)
