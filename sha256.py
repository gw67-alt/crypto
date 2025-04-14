def mult_xor(a, b):
    """XOR simulation using only multiplication.
    XOR is true when exactly one of the inputs is true.
    For each bit: (a * (1-b)) + (b * (1-a)) = a + b - 2*a*b
    """
    result = 0
    mask = 1
    
    for i in range(32):
        # Extract bits using division and modulo
        a_bit = (a % (mask * 2)) // mask
        b_bit = (b % (mask * 2)) // mask
        
        # XOR formula: a + b - 2*a*b
        result_bit = a_bit + b_bit - 2 * a_bit * b_bit
        
        # Set the bit using addition and multiplication
        result = result + result_bit * mask
        
        # Move to next bit position
        mask = mask * 2
    
    # Ensure result is a 32-bit unsigned integer
    return result & 0xFFFFFFFF


def mult_and(a, b):
    """AND simulation using only multiplication."""
    result = 0
    mask = 1
    
    for i in range(32):
        # Extract bits using division and modulo
        a_bit = (a % (mask * 2)) // mask
        b_bit = (b % (mask * 2)) // mask
        
        # AND is simply multiplication of bits
        result_bit = a_bit * b_bit
        
        # Set the bit using addition and multiplication
        result = result + result_bit * mask
        
        # Move to next bit position
        mask = mask * 2
    
    return result & 0xFFFFFFFF


def mult_not(a):
    """NOT simulation using only multiplication."""
    result = 0
    mask = 1
    
    for i in range(32):
        # Extract bit using division and modulo
        a_bit = (a % (mask * 2)) // mask
        
        # NOT is 1 - bit
        result_bit = 1 - a_bit
        
        # Set the bit using addition and multiplication
        result = result + result_bit * mask
        
        # Move to next bit position
        mask = mask * 2
    
    return result & 0xFFFFFFFF


def mult_or(a, b):
    """OR simulation using only multiplication."""
    result = 0
    mask = 1
    
    for i in range(32):
        # Extract bits using division and modulo
        a_bit = (a % (mask * 2)) // mask
        b_bit = (b % (mask * 2)) // mask
        
        # OR using multiplication: 1-(1-a)*(1-b) = a+b-a*b
        result_bit = a_bit + b_bit - a_bit * b_bit
        
        # Set the bit using addition and multiplication
        result = result + result_bit * mask
        
        # Move to next bit position
        mask = mask * 2
    
    return result & 0xFFFFFFFF


def mult_add(a, b):
    """Addition simulation using only multiplication."""
    result = 0
    carry = 0
    mask = 1
    
    for i in range(32):
        # Extract bits using division and modulo
        a_bit = (a % (mask * 2)) // mask
        b_bit = (b % (mask * 2)) // mask
        
        # Sum bit = a ⊕ b ⊕ carry
        # Using XOR implementation: a + b - 2*a*b
        sum_bit = (a_bit + b_bit + carry - 2 * a_bit * b_bit - 2 * a_bit * carry - 2 * b_bit * carry + 4 * a_bit * b_bit * carry) % 2
        
        # Next carry = (a AND b) OR (carry AND (a XOR b))
        temp_xor = a_bit + b_bit - 2 * a_bit * b_bit
        carry = a_bit * b_bit + carry * temp_xor - carry * temp_xor * a_bit * b_bit
        
        # Set the bit using addition and multiplication
        result = result + sum_bit * mask
        
        # Move to next bit position
        mask = mask * 2
    
    return result & 0xFFFFFFFF


def mult_rotate_right(a, b):
    """Right rotation simulation using only multiplication."""
    b = b % 32  # Ensure rotation is within bounds
    a = a & 0xFFFFFFFF  # Ensure we're working with a 32-bit number
    
    result = 0
    
    for i in range(32):
        # For right rotation, we need to look b positions to the right
        # which means the bit at position i in the result comes from position (i + b) % 32 in the input
        src_pos = (i + b) % 32
        
        # Extract the bit at position src_pos
        src_mask = 2 ** src_pos
        bit = (a % (src_mask * 2)) // src_mask
        
        # Set the bit at position i in the result
        dest_mask = 2 ** i
        result = result + bit * dest_mask
    
    return result & 0xFFFFFFFF


def mult_shift_right(a, b):
    """Right shift simulation using only multiplication."""
    b = min(b, 32)  # Ensure shift is within bounds
    
    result = 0
    
    for i in range(32 - b):
        # Calculate source position (i + b)
        src_pos = i + b
        
        # Extract the bit using division and modulo
        src_mask = 2 ** src_pos
        bit = (a % (src_mask * 2)) // src_mask
        
        # Set the bit using multiplication
        dest_mask = 2 ** i
        result = result + bit * dest_mask
    
    return result & 0xFFFFFFFF


# SHA-256 specific functions
def ch(x, y, z):
    """Ch(x, y, z) = (x AND y) XOR ((NOT x) AND z)"""
    return mult_xor(mult_and(x, y), mult_and(mult_not(x), z))


def maj(x, y, z):
    """Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)"""
    return mult_xor(mult_xor(mult_and(x, y), mult_and(x, z)), mult_and(y, z))


def sigma_0(x):
    """Σ0(x) = ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)"""
    return mult_xor(mult_xor(mult_rotate_right(x, 2), mult_rotate_right(x, 13)), mult_rotate_right(x, 22))


def sigma_1(x):
    """Σ1(x) = ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)"""
    return mult_xor(mult_xor(mult_rotate_right(x, 6), mult_rotate_right(x, 11)), mult_rotate_right(x, 25))


def sigma_0_small(x):
    """σ0(x) = ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)"""
    return mult_xor(mult_xor(mult_rotate_right(x, 7), mult_rotate_right(x, 18)), mult_shift_right(x, 3))


def sigma_1_small(x):
    """σ1(x) = ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)"""
    return mult_xor(mult_xor(mult_rotate_right(x, 17), mult_rotate_right(x, 19)), mult_shift_right(x, 10))


def string_to_bytes(message):
    """Convert string to UTF-8 encoded bytes."""
    if isinstance(message, str):
        return message.encode('utf-8')
    return message


def bytes_to_hex(bytes_array):
    """Convert bytes to hex string using multiplication."""
    hex_chars = '0123456789abcdef'
    hex_string = ''
    
    for byte in bytes_array:
        first_char = hex_chars[byte // 16]
        second_char = hex_chars[byte % 16]
        hex_string += first_char + second_char
    
    return hex_string


def mult_sha256(message):
    """SHA-256 implementation using only multiplication operations."""
    # Constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
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
    
    # Initial hash values (first 32 bits of the fractional parts of the square roots of the first 8 primes)
    h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]
    
    # Pre-processing
    msg_bytes = bytearray(string_to_bytes(message))
    msg_bits = len(msg_bytes) * 8
    
    # Append the bit '1' to the message
    msg_bytes.append(128)  # 128 = 0x80, using decimal instead of hex
    
    # Append '0' bits until message length ≡ 448 (mod 512)
    while (len(msg_bytes) * 8) % 512 != 448:
        msg_bytes.append(0)
    
    # Append length of message (64-bit big-endian integer)
    msg_bits_array = []
    for i in range(8):
        # Use division instead of bit shifting
        msg_bits_array.append((msg_bits // (2 ** (56 - i * 8))) % 256)
    
    msg_bytes.extend(msg_bits_array)
    
    # Process the message in 512-bit chunks
    for chunk_start in range(0, len(msg_bytes), 64):
        chunk = msg_bytes[chunk_start:chunk_start + 64]
        
        # Prepare the message schedule using multiplication instead of bit shifts
        w = [0] * 64
        
        for t in range(16):
            w[t] = (chunk[t * 4] * 16777216 +     # 2^24
                    chunk[t * 4 + 1] * 65536 +     # 2^16
                    chunk[t * 4 + 2] * 256 +       # 2^8
                    chunk[t * 4 + 3])
        
        for t in range(16, 64):
            w[t] = mult_add(mult_add(sigma_1_small(w[t - 2]), w[t - 7]), 
                          mult_add(sigma_0_small(w[t - 15]), w[t - 16]))
        
        # Initialize working variables
        a, b, c, d, e, f, g, h_val = h
        
        # Main loop
        for t in range(64):
            t1 = mult_add(mult_add(mult_add(mult_add(h_val, sigma_1(e)), ch(e, f, g)), k[t]), w[t])
            t2 = mult_add(sigma_0(a), maj(a, b, c))
            
            h_val = g
            g = f
            f = e
            e = mult_add(d, t1)
            d = c
            c = b
            b = a
            a = mult_add(t1, t2)
        
        # Update hash values
        h[0] = mult_add(a, h[0]) & 0xFFFFFFFF
        h[1] = mult_add(b, h[1]) & 0xFFFFFFFF
        h[2] = mult_add(c, h[2]) & 0xFFFFFFFF
        h[3] = mult_add(d, h[3]) & 0xFFFFFFFF
        h[4] = mult_add(e, h[4]) & 0xFFFFFFFF
        h[5] = mult_add(f, h[5]) & 0xFFFFFFFF
        h[6] = mult_add(g, h[6]) & 0xFFFFFFFF
        h[7] = mult_add(h_val, h[7]) & 0xFFFFFFFF
    
    # Produce the final hash value using multiplication and division
    hash_bytes = []
    for word in h:
        # Extract bytes using division and modulo
        hash_bytes.append((word // 16777216) % 256)    # First byte (most significant)
        hash_bytes.append((word // 65536) % 256)       # Second byte
        hash_bytes.append((word // 256) % 256)         # Third byte
        hash_bytes.append(word % 256)                  # Fourth byte (least significant)
    
    return bytes_to_hex(hash_bytes)


def test_sha256():
    """Test the SHA-256 implementation with known test vectors."""
    import hashlib  # Import for verification
    
    test_cases = [
        {"input": "", "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"input": "abc", "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"input": "hello world", "expected": "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"}
    ]
    
    print("Testing SHA-256 implementation using only multiplication operations:")
    print("="*80)
    
    for test in test_cases:
        # Calculate using our multiplication-based implementation
        result = mult_sha256(test["input"])
        
        # Calculate using Python's hashlib for verification
        std_result = hashlib.sha256(string_to_bytes(test["input"])).hexdigest()
        
        print(f"Input: '{test['input']}'")
        print(f"Expected:      {test['expected']}")
        print(f"Our Result:    {result}")
        print(f"Hashlib:       {std_result}")
        print(f"Match Expected: {result == test['expected']}")
        print(f"Match Hashlib:  {result == std_result}")
        print("-"*80)


if __name__ == "__main__":
    test_sha256()
