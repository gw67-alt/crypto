import time

def mult_sha256(message):
    """
    SHA-256 implementation using primarily multiplication operations
    instead of bitwise operations where possible.
    """
    # SHA-256 Constants
    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    # Initial hash values
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19
    
    # ----- Multiplication-based Bit Operations -----
    
    def mult_right_shift(x, n):
        """Right shift using division by powers of 2"""
        return int(x // (2 ** n))
    
    def mult_left_shift(x, n):
        """Left shift using multiplication by powers of 2"""
        return int(x * (2 ** n)) & 0xFFFFFFFF
    
    def mult_and(a, b):
        """
        Implement AND using multiplication
        For binary values, a AND b = a * b
        Need to process bit by bit for multi-bit values
        """
        result = 0
        mask = 1
        
        for _ in range(32):  # 32-bit integers
            bit_a = 1 if (a & mask) != 0 else 0
            bit_b = 1 if (b & mask) != 0 else 0
            
            # AND is simply a * b for binary bits
            bit_result = (bit_a * bit_b) * mask
            
            # Add to result
            result += bit_result
            
            # Move to next bit
            mask *= 2
            
        return result
    
    def mult_or(a, b):
        """
        Implement OR using multiplication and addition
        For binary values, a OR b = a + b - (a * b)
        Need to process bit by bit for multi-bit values
        """
        result = 0
        mask = 1
        
        for _ in range(32):  # 32-bit integers
            bit_a = 1 if (a & mask) != 0 else 0
            bit_b = 1 if (b & mask) != 0 else 0
            
            # OR formula: a + b - (a * b) for binary bits
            bit_result = (bit_a + bit_b - (bit_a * bit_b)) * mask
            
            # Add to result
            result += bit_result
            
            # Move to next bit
            mask *= 2
            
        return result
    
    def mult_xor(a, b):
        """
        Implement XOR using multiplication, addition, and subtraction
        For binary values, a XOR b = a + b - 2*(a * b)
        Need to process bit by bit for multi-bit values
        """
        result = 0
        mask = 1
        
        for _ in range(32):  # 32-bit integers
            bit_a = 1 if (a & mask) != 0 else 0
            bit_b = 1 if (b & mask) != 0 else 0
            
            # XOR formula: a + b - 2*(a * b) for binary bits
            bit_result = (bit_a + bit_b - 2 * (bit_a * bit_b)) * mask
            
            # Add to result
            result += bit_result
            
            # Move to next bit
            mask *= 2
            
        return result & 0xFFFFFFFF
    
    def mult_not(a):
        """
        Implement NOT using subtraction
        For binary values, NOT a = 1 - a
        Need to process bit by bit for multi-bit values
        """
        result = 0
        mask = 1
        
        for _ in range(32):  # 32-bit integers
            bit_a = 1 if (a & mask) != 0 else 0
            
            # NOT formula: 1 - a for binary bits
            bit_result = (1 - bit_a) * mask
            
            # Add to result
            result += bit_result
            
            # Move to next bit
            mask *= 2
            
        return result & 0xFFFFFFFF
    
    def mult_rotate_right(x, n):
        """
        Right rotation using multiplication-based shifts
        """
        right_part = mult_right_shift(x, n)
        left_part = mult_left_shift(x, (32 - n))
        return mult_or(right_part, left_part) & 0xFFFFFFFF
    
    # ----- Message Preprocessing -----
    
    def preprocessing(message):
        """Convert message to padded bit array"""
        # Convert message to binary
        if isinstance(message, str):
            message = message.encode()
        
        # Convert bytes to a list of bits
        bits = []
        for byte in message:
            for i in range(8):
                bits.append(mult_right_shift(byte, (7 - i)) & 1)
        
        # Original message length in bits
        original_length = len(bits)
        
        # Append the bit '1' to the message
        bits.append(1)
        
        # Append 0s until the length in bits ≡ 448 (mod 512)
        while len(bits) % 512 != 448:
            bits.append(0)
        
        # Append 64 bits representing the original length
        for i in range(64):
            bits.append(mult_right_shift(original_length, (63 - i)) & 1)
        
        return bits
    
    # ----- Main SHA-256 Algorithm -----
    
    # Process the message
    bits = preprocessing(message)
    
    # Divide into 512-bit chunks
    chunks = [bits[i:i + 512] for i in range(0, len(bits), 512)]
    
    # Process each chunk
    for chunk in chunks:
        # Create message schedule
        w = []
        
        # Break chunk into sixteen 32-bit words
        for i in range(16):
            word = 0
            for j in range(32):
                word = mult_left_shift(word, 1) | chunk[i * 32 + j]
            w.append(word)
        
        # Extend to 64 words
        for i in range(16, 64):
            # Use multiplication-based operations
            s0 = mult_xor(mult_rotate_right(w[i-15], 7), 
                         mult_xor(mult_rotate_right(w[i-15], 18), 
                                 mult_right_shift(w[i-15], 3)))
            
            s1 = mult_xor(mult_rotate_right(w[i-2], 17), 
                         mult_xor(mult_rotate_right(w[i-2], 19), 
                                 mult_right_shift(w[i-2], 10)))
            
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
        
        # Compression function main loop
        for i in range(64):
            # Use multiplication-based operations
            S1 = mult_xor(mult_rotate_right(e, 6), 
                         mult_xor(mult_rotate_right(e, 11), 
                                 mult_rotate_right(e, 25)))
            
            # ch = (e & f) ^ ((~e) & g)
            ch = mult_xor(mult_and(e, f), mult_and(mult_not(e), g))
            
            temp1 = (h + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF
            
            S0 = mult_xor(mult_rotate_right(a, 2), 
                         mult_xor(mult_rotate_right(a, 13), 
                                 mult_rotate_right(a, 22)))
            
            # maj = (a & b) ^ (a & c) ^ (b & c)
            maj = mult_xor(mult_and(a, b), 
                          mult_xor(mult_and(a, c), 
                                  mult_and(b, c)))
            
            temp2 = (S0 + maj) & 0xFFFFFFFF
            
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF
        
        # Add compressed chunk to hash values
        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = (h4 + e) & 0xFFFFFFFF
        h5 = (h5 + f) & 0xFFFFFFFF
        h6 = (h6 + g) & 0xFFFFFFFF
        h7 = (h7 + h) & 0xFFFFFFFF
    
    # Produce the final hash value
    digest = ''
    for val in [h0, h1, h2, h3, h4, h5, h6, h7]:
        digest += f'{val:08x}'
    
    return digest

# Standard SHA-256 implementation for comparison
def standard_sha256(message):
    """Standard SHA-256 using built-in operators for comparison"""
    # SHA-256 Constants
    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    # Initial hash values
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19
    
    def right_rotate(x, y):
        """Rotate x right by y bits"""
        return ((x >> y) | (x << (32 - y))) & 0xFFFFFFFF
    
    def preprocessing(message):
        """Convert message to padded bit array"""
        # Convert message to binary
        if isinstance(message, str):
            message = message.encode()
        
        # Convert bytes to a list of bits
        bits = []
        for byte in message:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        # Original message length in bits
        original_length = len(bits)
        
        # Append the bit '1' to the message
        bits.append(1)
        
        # Append 0s until the length in bits ≡ 448 (mod 512)
        while len(bits) % 512 != 448:
            bits.append(0)
        
        # Append 64 bits representing the original length
        for i in range(64):
            bits.append((original_length >> (63 - i)) & 1)
        
        return bits
    
    # Process the message
    bits = preprocessing(message)
    
    # Divide into 512-bit chunks
    chunks = [bits[i:i + 512] for i in range(0, len(bits), 512)]
    
    # Process each chunk
    for chunk in chunks:
        # Create message schedule
        w = []
        
        # Break chunk into sixteen 32-bit words
        for i in range(16):
            word = 0
            for j in range(32):
                word = (word << 1) | chunk[i * 32 + j]
            w.append(word)
        
        # Extend to 64 words
        for i in range(16, 64):
            s0 = right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
        
        # Compression function main loop
        for i in range(64):
            S1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF
            S0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF
            
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF
        
        # Add compressed chunk to hash values
        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = (h4 + e) & 0xFFFFFFFF
        h5 = (h5 + f) & 0xFFFFFFFF
        h6 = (h6 + g) & 0xFFFFFFFF
        h7 = (h7 + h) & 0xFFFFFFFF
    
    # Produce the final hash value
    digest = ''
    for val in [h0, h1, h2, h3, h4, h5, h6, h7]:
        digest += f'{val:08x}'
    
    return digest

if __name__ == "__main__":
    # Test with a small string first to verify correctness
    test_message = "A" * 80
    
    # Calculate hash with both implementations
    start_time = time.time()
    standard_hash = standard_sha256(test_message)
    standard_time = time.time() - start_time
    
    start_time = time.time()
    mult_hash = mult_sha256(test_message)
    mult_time = time.time() - start_time
    
    # Print results
    print(f"Test message: '{test_message}'")
    print(f"Standard SHA-256: {standard_hash}")
    print(f"Multiplication SHA-256: {mult_hash}")
    print(f"Hashes match: {standard_hash == mult_hash}")
    print(f"Standard time: {standard_time:.6f} seconds")
    print(f"Multiplication time: {mult_time:.6f} seconds")
    print(f"Factor: {mult_time / standard_time:.2f}x")
    
    # Count multiplication operations
    mult_count = 0
    
    # Each mult_and, mult_or, mult_xor, mult_not operation
    # Analyze the S0, S1, ch, and maj calculations in one iteration:
    # - S0: 3 mult_rotate_right (each with 64 multiplications) + 2 mult_xor (each with 96 multiplications)
    # - S1: Same as S0
    # - ch: 1 mult_and (32 multiplications) + 1 mult_not (32 multiplications) + 1 mult_and (32 multiplications) + 1 mult_xor (96 multiplications)
    # - maj: 3 mult_and (96 multiplications) + 2 mult_xor (192 multiplications)
    
    # Simplified estimate of multiplications in one compression function iteration
    per_iteration_mults = 2*(3*64 + 2*96) + (32 + 32 + 32 + 96) + (96 + 192) 
    
    # For a small message with 1 chunk and 64 iterations
    total_mults = 64 * per_iteration_mults
    
    print(f"\nEstimated multiplications in one compression iteration: {per_iteration_mults}")
    print(f"Estimated total multiplications for small message: {total_mults}")
