def decimal_xor(a, b):
    """XOR simulation using only decimal operations."""
    result = 0
    power = 1
    
    for i in range(32):
        a_digit = (a // power) % 2
        b_digit = (b // power) % 2
        result_digit = a_digit + b_digit - 2 * a_digit * b_digit
        result = result + result_digit * power
        power = power * 2
    
    return result % 4294967296


def decimal_and(a, b):
    """AND simulation using only decimal operations."""
    result = 0
    power = 1
    
    for i in range(32):
        a_digit = (a // power) % 2
        b_digit = (b // power) % 2
        result_digit = a_digit * b_digit
        result = result + result_digit * power
        power = power * 2
    
    return result % 4294967296


def decimal_not(a):
    """NOT simulation using only decimal operations."""
    return (4294967295 - a) % 4294967296


def decimal_or(a, b):
    """OR simulation using only decimal operations."""
    result = 0
    power = 1
    
    for i in range(32):
        a_digit = (a // power) % 2
        b_digit = (b // power) % 2
        result_digit = a_digit + b_digit - a_digit * b_digit
        result = result + result_digit * power
        power = power * 2
    
    return result % 4294967296


def decimal_add(a, b):
    """Addition constrained to 32-bit unsigned integers."""
    return (a + b) % 4294967296


def decimal_rotate_right(a, n):
    """Right rotation simulation using only decimal operations."""
    n = n % 32
    mask = (1 << n) - 1
    return ((a & mask) * (1 << (32 - n)) + (a >> n)) % 4294967296


def decimal_shift_right(a, n):
    """Right shift simulation using only decimal operations."""
    if n >= 32:
        return 0
    return (a // (1 << n)) % 4294967296


# SHA-256 specific functions
def decimal_ch(x, y, z):
    """Ch(x, y, z) = (x AND y) XOR ((NOT x) AND z)"""
    return decimal_xor(decimal_and(x, y), decimal_and(decimal_not(x), z))


def decimal_maj(x, y, z):
    """Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)"""
    return decimal_xor(decimal_xor(decimal_and(x, y), decimal_and(x, z)), decimal_and(y, z))


def decimal_sigma_0(x):
    """Σ0(x) = ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)"""
    return decimal_xor(decimal_xor(decimal_rotate_right(x, 2), decimal_rotate_right(x, 13)), decimal_rotate_right(x, 22))


def decimal_sigma_1(x):
    """Σ1(x) = ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)"""
    return decimal_xor(decimal_xor(decimal_rotate_right(x, 6), decimal_rotate_right(x, 11)), decimal_rotate_right(x, 25))


def decimal_sigma_0_small(x):
    """σ0(x) = ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)"""
    return decimal_xor(decimal_xor(decimal_rotate_right(x, 7), decimal_rotate_right(x, 18)), decimal_shift_right(x, 3))


def decimal_sigma_1_small(x):
    """σ1(x) = ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)"""
    return decimal_xor(decimal_xor(decimal_rotate_right(x, 17), decimal_rotate_right(x, 19)), decimal_shift_right(x, 10))


def string_to_bytes(message):
    """Convert string to UTF-8 encoded bytes."""
    if isinstance(message, str):
        return message.encode('utf-8')
    return message


def bytes_to_hex(bytes_array):
    """Convert bytes to hex string using decimal operations."""
    hex_chars = '0123456789abcdef'
    hex_string = ''
    
    for byte in bytes_array:
        first_char = hex_chars[byte // 16]
        second_char = hex_chars[byte % 16]
        hex_string += first_char + second_char
    
    return hex_string


def decimal_sha256_with_sum_tracking(message):
    """SHA-256 implementation using decimal operations with memory sum tracking."""
    # Memory array to store all data
    memory = [0] * 144
    
    # Memory map:
    # 0-63:   Constants (k)
    # 64-71:  Hash values (h)
    # 72-135: Message schedule (w)
    # 136:    a
    # 137:    b
    # 138:    c
    # 139:    d
    # 140:    e
    # 141:    f
    # 142:    g
    # 143:    h
    
    # Initialize constants in memory[0:64]
    constants = [
        1116352408, 1899447441, 3049323471, 3921009573, 961987163, 1508970993, 2453635748, 2870763221,
        3624381080, 310598401, 607225278, 1426881987, 1925078388, 2162078206, 2614888103, 3248222580,
        3835390401, 4022224774, 264347078, 604807628, 770255983, 1249150122, 1555081692, 1996064986,
        2554220882, 2821834349, 2952996808, 3210313671, 3336571891, 3584528711, 113926993, 338241895,
        666307205, 773529912, 1294757372, 1396182291, 1695183700, 1986661051, 2177026350, 2456956037,
        2730485921, 2820302411, 3259730800, 3345764771, 3516065817, 3600352804, 4094571909, 275423344,
        430227734, 506948616, 659060556, 883997877, 958139571, 1322822218, 1537002063, 1747873779,
        1955562222, 2024104815, 2227730452, 2361852424, 2428436474, 2756734187, 3204031479, 3329325298
    ]
    
    print("=== Memory Sum Tracking ===")
    print("Initial sum (zeros): 0")
    
    # Load constants and track sum
    for i in range(64):
        memory[i] = constants[i]
    
    constants_sum = sum(memory)
    print(f"After loading constants (0-63): {constants_sum}")
    
    # Initialize hash values in memory[64:72]
    initial_hash = [
        1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225
    ]
    
    for i in range(8):
        memory[64 + i] = initial_hash[i]
    
    initial_sum = sum(memory)
    print(f"After loading initial hash values (64-71): {initial_sum}")
    print(f"Constants + Initial Hash Sum: {initial_sum}")
    
    # Pre-processing
    msg_bytes = bytearray(string_to_bytes(message))
    msg_length = len(msg_bytes) * 8  # Length in bits
    
    # For information only
    msg_bytes_sum = sum(msg_bytes)
    print(f"\nMessage: '{message}'")
    print(f"Message bytes: {list(msg_bytes)}")
    print(f"Sum of message bytes: {msg_bytes_sum}")
    
    # Append the bit '1' to the message
    msg_bytes.append(128)  # 10000000 in binary
    
    # Append '0' bits until message length ≡ 448 (mod 512)
    while (len(msg_bytes) * 8) % 512 != 448:
        msg_bytes.append(0)
    
    # Append original message length as 64-bit big-endian integer
    msg_length_bytes = [(msg_length >> (8 * i)) & 0xff for i in range(7, -1, -1)]
    msg_bytes.extend(msg_length_bytes)
    
    # For information only
    padded_msg_bytes_sum = sum(msg_bytes)
    print(f"Padded message bytes: {list(msg_bytes)}")
    print(f"Sum of padded message bytes: {padded_msg_bytes_sum}")
    
    # Track memory sums for each chunk
    chunk_index = 0
    
    # Process the message in 512-bit chunks (64 bytes)
    for chunk_start in range(0, len(msg_bytes), 64):
        chunk = msg_bytes[chunk_start:chunk_start + 64]
        chunk_index += 1
        print(f"\n--- Processing Chunk {chunk_index} ---")
        
        # Prepare the message schedule (w) in memory[72:136]
        for t in range(16):
            # Convert 4 bytes to a 32-bit word using decimal operations
            memory[72 + t] = (chunk[t * 4] * 16777216 +     # 2^24
                             chunk[t * 4 + 1] * 65536 +     # 2^16
                             chunk[t * 4 + 2] * 256 +       # 2^8
                             chunk[t * 4 + 3])
        
        schedule_sum_initial = sum(memory[72:88])
        print(f"Initial message schedule (w) sum (first 16 words): {schedule_sum_initial}")
        memory_sum = sum(memory)
        print(f"Memory sum after loading first 16 words: {memory_sum}")
        
        # Extend the first 16 words to the remaining 48 words
        for t in range(16, 64):
            memory[72 + t] = decimal_add(
                decimal_add(
                    decimal_sigma_1_small(memory[72 + t - 2]), 
                    memory[72 + t - 7]
                ),
                decimal_add(
                    decimal_sigma_0_small(memory[72 + t - 15]),
                    memory[72 + t - 16]
                )
            )
        
        full_schedule_sum = sum(memory[72:136])
        print(f"Full message schedule (w) sum (all 64 words): {full_schedule_sum}")
        memory_sum = sum(memory)
        print(f"Memory sum after extending message schedule: {memory_sum}")
        
        # Initialize working variables in memory[136:144]
        for i in range(8):
            memory[136 + i] = memory[64 + i]
        
        working_var_sum = sum(memory[136:144])
        print(f"Working variables (a-h) initial sum: {working_var_sum}")
        memory_sum = sum(memory)
        print(f"Memory sum after initializing working variables: {memory_sum}")
        
        # Track memory sum during compression
        round_sum_samples = []
        
        # Main compression loop
        for t in range(64):
            # T1 = h + Σ1(e) + Ch(e,f,g) + Kt + Wt
            t1 = decimal_add(
                decimal_add(
                    decimal_add(
                        decimal_add(
                            memory[143],  # h
                            decimal_sigma_1(memory[140])  # Σ1(e)
                        ),
                        decimal_ch(memory[140], memory[141], memory[142])  # Ch(e,f,g)
                    ),
                    memory[t]  # k[t]
                ),
                memory[72 + t]  # w[t]
            )
            
            # T2 = Σ0(a) + Maj(a,b,c)
            t2 = decimal_add(
                decimal_sigma_0(memory[136]),  # Σ0(a)
                decimal_maj(memory[136], memory[137], memory[138])  # Maj(a,b,c)
            )
            
            # Track temporary variables
            temp_sum = t1 + t2
            
            # Update working variables
            memory[143] = memory[142]  # h = g
            memory[142] = memory[141]  # g = f
            memory[141] = memory[140]  # f = e
            memory[140] = decimal_add(memory[139], t1)  # e = d + t1
            memory[139] = memory[138]  # d = c
            memory[138] = memory[137]  # c = b
            memory[137] = memory[136]  # b = a
            memory[136] = decimal_add(t1, t2)  # a = t1 + t2
            
            # Sample some rounds for sum tracking
            if t == 0 or t == 15 or t == 31 or t == 63:
                working_var_sum = sum(memory[136:144])
                memory_sum = sum(memory) + temp_sum  # Include t1 and t2
                round_sum_samples.append((t, working_var_sum, memory_sum))
        
        # Print sampled round sums
        print("\nCompression Round Samples:")
        for round_num, var_sum, mem_sum in round_sum_samples:
            print(f"Round {round_num}: Working Variables Sum = {var_sum}, Memory + Temp Sum = {mem_sum}")
        
        # Update hash values
        for i in range(8):
            memory[64 + i] = decimal_add(memory[136 + i], memory[64 + i])
        
        hash_sum = sum(memory[64:72])
        print(f"\nHash values (h) sum after chunk {chunk_index}: {hash_sum}")
        memory_sum = sum(memory)
        print(f"Final memory sum after chunk {chunk_index}: {memory_sum}")
    
    # Final memory sum
    final_memory_sum = sum(memory)
    print(f"\n=== Final Memory Sum ===")
    print(f"Total sum of all decimal integers: {final_memory_sum}")
    
    # Memory segment sums
    constants_sum = sum(memory[0:64])
    hash_sum = sum(memory[64:72])
    schedule_sum = sum(memory[72:136])
    working_sum = sum(memory[136:144])
    
    print(f"\nMemory Segment Sums:")
    print(f"Constants (k): {constants_sum}")
    print(f"Hash values (h): {hash_sum}")
    print(f"Message schedule (w): {schedule_sum}")
    print(f"Working variables (a-h): {working_sum}")
    
    # Produce the final hash value
    hash_bytes = []
    for i in range(8):
        word = memory[64 + i]
        # Convert 32-bit words to bytes (4 bytes each)
        hash_bytes.append((word >> 24) & 0xff)
        hash_bytes.append((word >> 16) & 0xff)
        hash_bytes.append((word >> 8) & 0xff)
        hash_bytes.append(word & 0xff)
    
    return bytes_to_hex(hash_bytes)


def test_decimal_sha256_with_sum():
    """Test the decimal-based SHA-256 implementation with memory sum tracking."""
    import hashlib  # Import for verification
    
    # Define test cases
    test_cases = [
        {"input": "", "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"input": "abc", "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"input": "hello world", "expected": "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"}
    ]
    
    # Test each case
    for test in test_cases:
        print("\n" + "="*80)
        print(f"Testing SHA-256 with input: '{test['input']}'")
        print("="*80)
        
        # Calculate using our decimal-based implementation with sum tracking
        result = decimal_sha256_with_sum_tracking(test["input"])
        
        # Calculate using Python's hashlib for verification
        std_result = hashlib.sha256(string_to_bytes(test["input"])).hexdigest()
        
        print("\n=== Results ===")
        print(f"Expected:      {test['expected']}")
        print(f"Our Result:    {result}")
        print(f"Hashlib:       {std_result}")
        print(f"Match Expected: {result == test['expected']}")
        print(f"Match Hashlib:  {result == std_result}")
        print("="*80 + "\n")


if __name__ == "__main__":
    test_decimal_sha256_with_sum()
