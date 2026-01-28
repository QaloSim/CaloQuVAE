import torch
import torch.nn as nn
import numpy as np

class GrayCodeOffset(nn.Module):
    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset

    def _gray_code(self, n):
        """Standard Gray code: n ^ (n >> 1)"""
        return n ^ (n >> 1)

    def _gray_to_binary(self, g):
        """
        Parallel Prefix XOR to convert Gray to Binary in O(log N).
        Supports integers up to 32 bits.
        """
        m = g.clone()
        m = m ^ (m >> 1)
        m = m ^ (m >> 2)
        m = m ^ (m >> 4)
        m = m ^ (m >> 8)
        m = m ^ (m >> 16)
        return m

    def encode(self, x, data_bits):
        """
        Encodes integer x into Gray Code Bits with Offset.
        
        Args:
            x: Int Tensor (Batch)
            data_bits: The resolution of the input data (e.g., 19). 
                       Output will be data_bits + 1.
        """
        out_bits = data_bits + 1
        
        # 1. Clamp inputs to valid data range (0 to 2^N - 1)
        #    This prevents overflowing into the offset region unpredictably.
        max_val = (1 << data_bits) - 1
        x_clamped = x.clamp(0, max_val)

        # 2. Apply Offset
        #    Shift the range to maximize Hamming distance at the boundaries.
        val = x_clamped + self.offset
        
        # 3. Compute Gray Code Integer
        gray_val = self._gray_code(val)
        
        # 4. Integer -> Bits (MSB first)
        #    Creates a mask: [2^19, 2^18, ... 1]
        mask = 2 ** torch.arange(out_bits - 1, -1, -1, device=x.device)
        
        #    Broadcast & Bitwise AND
        #    (Batch, 1) & (out_bits) -> (Batch, out_bits)
        bits = (gray_val.unsqueeze(-1) & mask).ne(0).float()
        
        return bits

    def decode_section(self, bit_tensor):
        """
        Reverses the encoding: Bits -> Gray -> Binary -> Subtract Offset
        """
        B, out_bits = bit_tensor.shape
        device = bit_tensor.device
        
        # 1. Bits -> Gray Integer
        powers = 2 ** torch.arange(out_bits - 1, -1, -1, device=device)
        gray_val = (bit_tensor * powers).sum(dim=1).int()
        
        # 2. Gray Integer -> Binary Integer
        val = self._gray_to_binary(gray_val)
        
        # 3. Remove Offset
        return (val - self.offset).float()

    def decode(self, x_encoded, lin_bits=19, sqrt_bits=16, log_bits=15):
        """
        Full decoding pipeline for the Energy tensor.
        Automatically handles the +1 bit expansion for slicing.
        """
        # Calculate actual slice widths (Resolution + 1)
        w_lin = lin_bits + 1
        w_sqrt = sqrt_bits + 1
        w_log = log_bits + 1
        
        # Slice
        lin_part = x_encoded[:, :w_lin]
        sqrt_part = x_encoded[:, w_lin : w_lin + w_sqrt]
        log_part = x_encoded[:, w_lin + w_sqrt : w_lin + w_sqrt + w_log]
        
        # Decode Linear
        E_lin = self.decode_section(lin_part)
        
        # Decode Sqrt
        sqrt_int = self.decode_section(sqrt_part)
        E_sqrt = (sqrt_int / 200.0).pow(2)
        
        # Decode Log
        log_int = self.decode_section(log_part)
        E_log = (log_int / 1e4).exp()
        
        return E_lin, E_sqrt, E_log