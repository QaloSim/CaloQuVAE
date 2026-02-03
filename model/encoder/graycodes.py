import torch
import torch.nn as nn
import numpy as np

class GrayCode(nn.Module):
    def __init__(self):
        super().__init__()
        # No offset storage required for standard Gray code

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
        Encodes integer x into Standard Gray Code Bits.
        
        Args:
            x: Int Tensor (Batch) or (Batch, 1)
            data_bits: The resolution of the input data. 
            
            outputs: (Batch, data_bits) float Tensor of bits
        """
        out_bits = data_bits
        
        # 1. Clamp inputs
        max_val = (1 << data_bits) - 1
        x_clamped = x.clamp(0, max_val)

        # 2. Compute Gray Code Integer
        gray_val = self._gray_code(x_clamped)
        
        # 3. Integer -> Bits (MSB first)
        #    Ensure gray_val is (Batch, 1)
        gray_val = gray_val.view(-1, 1)
        
        #    Create mask: [2^(N-1), ... 1]
        mask = 2 ** torch.arange(out_bits - 1, -1, -1, device=x.device)
        
        #    Broadcast: (Batch, 1) & (out_bits) -> (Batch, out_bits)
        #    Note: We do NOT unsqueeze gray_val here, because we already viewed it as (-1, 1).
        #    PyTorch will broadcast the 1D mask to (1, out_bits) automatically against the (Batch, 1) input.
        bits = (gray_val & mask).ne(0).float()
        
        return bits
    def decode_section(self, bit_tensor):
        """
        Reverses the encoding: Bits -> Gray -> Binary
        """
        B, out_bits = bit_tensor.shape
        device = bit_tensor.device
        
        # 1. Bits -> Gray Integer
        powers = 2 ** torch.arange(out_bits - 1, -1, -1, device=device)
        gray_val = (bit_tensor * powers).sum(dim=1).int()
        
        # 2. Gray Integer -> Binary Integer
        val = self._gray_to_binary(gray_val)
        
        # 3. Return Value (No offset subtraction)
        return val.float()

    def decode(self, x_encoded, lin_bits=19, sqrt_bits=17, log_bits=16):
        """
        Full decoding pipeline for the Energy tensor.
        Slices strictly by the requested bit widths (no +1 expansion).
        """
        # Calculate actual slice widths (Strictly equal to resolution)
        w_lin = lin_bits
        w_sqrt = sqrt_bits
        w_log = log_bits
        
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
        # E_log = ((log_int / 5e3)+2**15).exp()
        E_log = (log_int / 5e3).exp()
        
        return E_lin, E_sqrt, E_log