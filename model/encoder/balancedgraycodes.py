import torch
import torch.nn as nn
import hashlib

class BalancedGrayCodeCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ParameterDict() 
        # We need to store the permutations to reverse them later
        self.perms = {} 
        self.inv_perms = {}

    def _get_or_create_table(self, bits, device):
        key = str(bits)
        if key in self.tables:
            return self.tables[key]
        
        # 1. Generate Gray Code Integers
        max_val = 2**bits
        indices = torch.arange(max_val, dtype=torch.int32)
        gray_ints = indices ^ (indices >> 1)
        
        # 2. Expand to Bits
        powers = 2**torch.arange(bits - 1, -1, -1)
        table = (gray_ints.unsqueeze(1) & powers).ne(0).float()

        # 3. Deterministic Shuffling (Balance)
        seed = int(hashlib.md5(str(bits).encode()).hexdigest(), 16) % (2**32)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        
        # Save permutation and its inverse
        perm = torch.randperm(bits, generator=g_cpu)
        inv_perm = torch.argsort(perm)
        
        self.perms[key] = perm.to(device)
        self.inv_perms[key] = inv_perm.to(device)
        
        # Apply Shuffle
        balanced_table = table[:, perm]
        
        p = nn.Parameter(balanced_table, requires_grad=False)
        self.tables[key] = p
        return p.to(device)

    def lookup(self, x_indices, bits):
        table = self._get_or_create_table(bits, x_indices.device)
        max_idx = table.shape[0] - 1
        indices = x_indices.long().clamp(0, max_idx)
        return table[indices]

    def _gray_to_binary_int(self, gray_val):
        """
        Parallel Prefix XOR algorithm to convert Gray to Binary in O(log N).
        Works for up to 32 bits.
        """
        b = gray_val.clone()
        b = b ^ (b >> 1)
        b = b ^ (b >> 2)
        b = b ^ (b >> 4)
        b = b ^ (b >> 8)
        b = b ^ (b >> 16)
        return b

    def decode_section(self, bit_tensor, bits):
        """
        Decodes a single chunk (e.g., just the linear bits).
        """
        device = bit_tensor.device
        key = str(bits)
        
        # Ensure table/perms exist on this device
        if key not in self.inv_perms:
            self._get_or_create_table(bits, device)
            
        # 1. Un-shuffle (Balance -> Standard Gray)
        # We index the columns using the inverse permutation
        inv_perm = self.inv_perms[key]
        standard_gray_bits = bit_tensor[:, inv_perm]
        
        # 2. Bits -> Integer (Standard Gray)
        powers = 2**torch.arange(bits - 1, -1, -1, device=device)
        gray_int = (standard_gray_bits * powers).sum(dim=1).int()
        
        # 3. Integer Gray -> Integer Binary (Fast XOR trick)
        binary_int = self._gray_to_binary_int(gray_int)
        
        return binary_int.float()

    def decode(self, x_encoded, lin_bits=19, sqrt_bits=17, log_bits=17):
        """
        Full decoding pipeline: Bits -> Physical Energy
        """
        # Split the concatenated tensor back into parts
        lin_part = x_encoded[:, :lin_bits]
        sqrt_part = x_encoded[:, lin_bits : lin_bits+sqrt_bits]
        log_part = x_encoded[:, lin_bits+sqrt_bits : lin_bits+sqrt_bits+log_bits]
        
        # Decode Linear
        lin_int = self.decode_section(lin_part, lin_bits)
        E_lin = lin_int # No scaling was applied
        
        # Decode Sqrt
        sqrt_int = self.decode_section(sqrt_part, sqrt_bits)
        E_sqrt = (sqrt_int / 200.0).pow(2)
        
        # Decode Log
        log_int = self.decode_section(log_part, log_bits)
        E_log = (log_int / 1e4).exp()
        
        return E_lin, E_sqrt, E_log