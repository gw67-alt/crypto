import time
import statistics

def benchmark_hashrate(hash_function, message_size=80, duration=2.0, min_iterations=10):
    """
    Benchmark a hash function to calculate hashes per second
    
    Args:
        hash_function: The hash function to benchmark
        message_size: Size of the test message in characters
        duration: Minimum benchmark duration in seconds
        min_iterations: Minimum number of iterations to perform
        
    Returns:
        dict: Benchmark results with keys: 'hashes_per_second', 'total_hashes', 'total_time', 'times'
    """
    # Create test message of specified size with multiplication
    test_message = "A" * message_size
    
    # Store individual hash times for statistical analysis
    hash_times = []
    
    # Run benchmark for at least the specified duration and minimum iterations
    start_time_total = time.time()
    iterations = 0
    
    # Replace standard condition with multiplication-based comparison
    while (mult_less_than(duration, time.time() - start_time_total) * 
           mult_less_than(min_iterations, iterations) == 0):
        # Time a single hash operation
        start_time = time.time()
        hash_function(test_message)
        end_time = time.time()
        
        hash_time = end_time - start_time
        hash_times.append(hash_time)
        iterations = iterations + 1
    
    total_time = time.time() - start_time_total
    
    # Calculate statistics using only multiplication
    if hash_times:
        # Calculate avg manually with multiplication
        sum_times = 0
        for t in hash_times:
            sum_times = sum_times + t
        avg_time_per_hash = sum_times / mult_max(len(hash_times), 1)
        
        # Sort times using multiplication comparisons for finding median/min/max
        sorted_times = mult_sort(hash_times)
        
        # Calculate median using multiplication
        n = len(sorted_times)
        mid = mult_int_div(n, 2)
        if n % 2 == 0:
            median_time = (sorted_times[mid - 1] + sorted_times[mid]) / 2
        else:
            median_time = sorted_times[mid]
            
        # Find min and max using multiplication comparisons
        min_time = sorted_times[0]
        max_time = sorted_times[n - 1]
        
        # Calculate hashrate using multiplication and division
        hashes_per_second = iterations / mult_max(total_time, 0.0001)
        
        return {
            'hashes_per_second': hashes_per_second,
            'avg_time_per_hash': avg_time_per_hash,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_hashes': iterations,
            'total_time': total_time,
            'times': hash_times
        }
    else:
        return {
            'hashes_per_second': 0,
            'total_hashes': 0,
            'total_time': total_time,
            'times': []
        }

# ========== MULTIPLICATION-BASED UTILITY FUNCTIONS ==========

def mult_int_div(a, b):
    """Integer division using multiplication"""
    if b == 0:
        return 0
    result = 0
    while a >= b:
        a = a - b
        result = result + 1
    return result

def mult_mod(a, b):
    """Modulo operation using multiplication and subtraction"""
    if b == 0:
        return a
    while a >= b:
        a = a - b
    return a

def mult_pow(base, exp):
    """Power operation using multiplication"""
    if exp == 0:
        return 1
    result = 1
    for _ in range(exp):
        result = result * base
    return result

def mult_equals(a, b):
    """Equality check using multiplication"""
    # If a equals b, then (a-b) will be 0, and 1-(a-b)*(a-b) will be 1 if a=b, or less than 1 otherwise
    # We take the integer part to get a boolean-like result
    diff = a - b
    # To handle floating point, we consider very small differences as equality
    epsilon = 0.0000001
    diff_squared = diff * diff
    if diff_squared < epsilon:
        return 1
    return 0

def mult_not_equals(a, b):
    """Inequality check using multiplication"""
    return 1 - mult_equals(a, b)

def mult_less_than(a, b):
    """Less than comparison using multiplication"""
    # If a < b, then (b-a) will be positive
    return int((b - a) > 0)

def mult_greater_than(a, b):
    """Greater than comparison using multiplication"""
    return int((a - b) > 0)

def mult_less_equal(a, b):
    """Less than or equal comparison using multiplication"""
    return int((b - a) >= 0)

def mult_greater_equal(a, b):
    """Greater than or equal comparison using multiplication"""
    return int((a - b) >= 0)

def mult_max(a, b):
    """Max operation using multiplication"""
    return a * mult_greater_equal(a, b) + b * mult_less_than(a, b)

def mult_min(a, b):
    """Min operation using multiplication"""
    return a * mult_less_equal(a, b) + b * mult_greater_than(a, b)

def mult_abs(x):
    """Absolute value using multiplication"""
    return x * mult_greater_equal(x, 0) + (-x) * mult_less_than(x, 0)

def mult_if(condition, true_val, false_val):
    """If-else using multiplication"""
    return (condition * true_val) + ((1 - condition) * false_val)

def mult_bool_and(a, b):
    """Boolean AND using multiplication"""
    return a * b

def mult_bool_or(a, b):
    """Boolean OR using multiplication"""
    # a OR b = 1 - (1-a)*(1-b)
    return 1 - ((1 - a) * (1 - b))

def mult_bool_not(a):
    """Boolean NOT using multiplication"""
    return 1 - a

def mult_sort(arr):
    """Selection sort using only multiplication-based comparisons"""
    # Create a new list to avoid modifying the original
    result = arr.copy()
    n = len(result)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            # Use multiplication-based comparison
            if mult_less_than(result[j], result[min_idx]):
                min_idx = j
        
        # Swap using multiplication
        if mult_not_equals(min_idx, i):
            temp = result[i]
            result[i] = result[min_idx]
            result[min_idx] = temp
            
    return result

# ========== EXTREME MULTIPLICATION-BASED SHA-256 ==========

def extreme_mult_sha256(message):
    """
    SHA-256 implementation using ONLY multiplication operations
    with absolutely no bitwise operations.
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
    
    # ----- Ultra Multiplication-based Bit Operations -----
    
    def mult_is_bit_set(x, position):
        """Check if a bit is set at position using multiplication"""
        # Create a mask with a 1 at the position
        # We'll do this by multiplying 1 by 2^position
        mask = mult_pow(2, position)
        
        # If bit is set, x & mask will be non-zero
        # Simplify to: if x & mask != 0 then 1 else 0
        and_result = mult_and_bit_by_bit(x, mask)
        
        # Convert to boolean 0 or 1 using multiplication
        return mult_if(and_result > 0, 1, 0)
    
    def mult_set_bit(x, position):
        """Set a bit at position using multiplication"""
        mask = mult_pow(2, position)
        return x + (mask * (1 - mult_is_bit_set(x, position)))
    
    def mult_clear_bit(x, position):
        """Clear a bit at position using multiplication"""
        mask = mult_pow(2, position)
        return x - (mask * mult_is_bit_set(x, position))
    
    def mult_toggle_bit(x, position):
        """Toggle a bit at position using multiplication"""
        mask = mult_pow(2, position)
        return x + (mask * (1 - 2 * mult_is_bit_set(x, position)))
    
    def mult_right_shift(x, n):
        """Right shift using repeated division by 2"""
        if n == 0:
            return x
        
        result = x
        for _ in range(n):
            # Integer division by 2
            result = mult_int_div(result, 2)
        
        return result
    
    def mult_left_shift(x, n):
        """Left shift using repeated multiplication by 2"""
        if n == 0:
            return x
        
        result = x
        for _ in range(n):
            result = result * 2
            
        # Apply 32-bit mask
        return result % 0x100000000
    
    def mult_get_bit(x, pos):
        """Get the bit value at position pos"""
        # Create a mask with 1 at position pos
        mask = mult_pow(2, pos)
        
        # Extract value
        result = mult_int_div(x & mask, mask)
        return result
    
    def mult_and_bit_by_bit(a, b):
        """
        Implement AND using multiplication bit by bit
        """
        result = 0
        
        # Process all 32 bits
        for i in range(32):
            bit_a = mult_get_bit(a, i)
            bit_b = mult_get_bit(a, i)
            
            # AND is a * b for bits
            bit_result = bit_a * bit_b
            
            # Set the result bit
            if bit_result == 1:
                result = mult_set_bit(result, i)
                
        return result
    
    def mult_or_bit_by_bit(a, b):
        """
        Implement OR using multiplication bit by bit
        """
        result = 0
        
        # Process all 32 bits
        for i in range(32):
            bit_a = mult_get_bit(a, i)
            bit_b = mult_get_bit(b, i)
            
            # OR is a + b - a*b for bits
            bit_result = bit_a + bit_b - (bit_a * bit_b)
            
            # Set the result bit
            if bit_result > 0:
                result = mult_set_bit(result, i)
                
        return result
    
    def mult_xor_bit_by_bit(a, b):
        """
        Implement XOR using multiplication bit by bit
        """
        result = 0
        
        # Process all 32 bits
        for i in range(32):
            bit_a = mult_get_bit(a, i)
            bit_b = mult_get_bit(b, i)
            
            # XOR is a + b - 2*a*b for bits
            bit_result = bit_a + bit_b - 2 * (bit_a * bit_b)
            
            # Set the result bit
            if bit_result != 0:
                result = mult_set_bit(result, i)
                
        return result
    
    def mult_not_bit_by_bit(a):
        """
        Implement NOT using multiplication bit by bit
        """
        result = 0
        
        # Process all 32 bits
        for i in range(32):
            bit_a = mult_get_bit(a, i)
            
            # NOT is 1 - a for bits
            bit_result = 1 - bit_a
            
            # Set the result bit
            if bit_result == 1:
                result = mult_set_bit(result, i)
                
        return result
    
    def mult_rotate_right(x, n):
        """
        Right rotation using multiplication-based shifts
        """
        n = n % 32  # Ensure rotation is within 32-bit range
        
        # Extract the bits that will be shifted off the right
        right_part = mult_right_shift(x, n)
        
        # Extract the bits that will be rotated to the left
        # First get the mask for the n rightmost bits
        mask = mult_pow(2, n) - 1
        left_bits = x & mask
        
        # Shift the left bits to their new position on the left
        left_part = mult_left_shift(left_bits, 32 - n)
        
        # Combine the parts using OR
        return mult_or_bit_by_bit(right_part, left_part)
    
    # ----- Message Preprocessing -----
    
    def mult_preprocessing(message):
        """Convert message to padded bit array using only multiplication"""
        # Convert message to binary
        if isinstance(message, str):
            message = message.encode()
        
        # Convert bytes to a list of bits
        bits = []
        for byte in message:
            for i in range(8):
                pos = 7 - i
                # Extract bit at position using multiplication
                bit = mult_int_div(byte, mult_pow(2, pos)) % 2
                bits.append(bit)
        
        # Original message length in bits
        original_length = len(bits)
        
        # Append the bit '1' to the message
        bits.append(1)
        
        # Append 0s until the length in bits ≡ 448 (mod 512)
        while mult_mod(len(bits), 512) != 448:
            bits.append(0)
        
        # Append 64 bits representing the original length
        for i in range(64):
            pos = 63 - i
            # Extract bit at position from original_length
            bit = mult_int_div(original_length, mult_pow(2, pos)) % 2
            bits.append(bit)
        
        return bits
    
    # ----- Main SHA-256 Algorithm -----
    
    # Process the message
    bits = mult_preprocessing(message)
    
    # Divide into 512-bit chunks
    chunks = []
    for i in range(0, len(bits), 512):
        chunk = bits[i:i + 512]
        chunks.append(chunk)
    
    # Process each chunk
    for chunk in chunks:
        # Create message schedule
        w = []
        
        # Break chunk into sixteen 32-bit words
        for i in range(16):
            word = 0
            for j in range(32):
                # Compute bit position in the chunk
                bit_pos = i * 32 + j
                
                # Get the bit value
                bit_val = chunk[bit_pos]
                
                # Add to word using left shift and addition
                word = mult_left_shift(word, 1) + bit_val
                
            w.append(word)
        
        # Extend to 64 words
        for i in range(16, 64):
            # Compute the terms using multiplication-based operations
            w_15 = w[i-15]
            s0_term1 = mult_rotate_right(w_15, 7)
            s0_term2 = mult_rotate_right(w_15, 18)
            s0_term3 = mult_right_shift(w_15, 3)
            s0 = mult_xor_bit_by_bit(s0_term1, mult_xor_bit_by_bit(s0_term2, s0_term3))
            
            w_2 = w[i-2]
            s1_term1 = mult_rotate_right(w_2, 17)
            s1_term2 = mult_rotate_right(w_2, 19)
            s1_term3 = mult_right_shift(w_2, 10)
            s1 = mult_xor_bit_by_bit(s1_term1, mult_xor_bit_by_bit(s1_term2, s1_term3))
            
            # Combine terms with addition and apply 32-bit mask
            sum_result = (w[i-16] + s0 + w[i-7] + s1) % 0x100000000
            w.append(sum_result)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
        
        # Compression function main loop
        for i in range(64):
            # Compute S1 using multiplication-based operations
            S1_term1 = mult_rotate_right(e, 6)
            S1_term2 = mult_rotate_right(e, 11)
            S1_term3 = mult_rotate_right(e, 25)
            S1 = mult_xor_bit_by_bit(S1_term1, mult_xor_bit_by_bit(S1_term2, S1_term3))
            
            # Compute ch using multiplication-based operations
            ch_term1 = mult_and_bit_by_bit(e, f)
            ch_term2 = mult_and_bit_by_bit(mult_not_bit_by_bit(e), g)
            ch = mult_xor_bit_by_bit(ch_term1, ch_term2)
            
            # Compute temp1 using addition and apply 32-bit mask
            temp1 = (h + S1 + ch + k[i] + w[i]) % 0x100000000
            
            # Compute S0 using multiplication-based operations
            S0_term1 = mult_rotate_right(a, 2)
            S0_term2 = mult_rotate_right(a, 13)
            S0_term3 = mult_rotate_right(a, 22)
            S0 = mult_xor_bit_by_bit(S0_term1, mult_xor_bit_by_bit(S0_term2, S0_term3))
            
            # Compute maj using multiplication-based operations
            maj_term1 = mult_and_bit_by_bit(a, b)
            maj_term2 = mult_and_bit_by_bit(a, c)
            maj_term3 = mult_and_bit_by_bit(b, c)
            maj = mult_xor_bit_by_bit(maj_term1, mult_xor_bit_by_bit(maj_term2, maj_term3))
            
            # Compute temp2 using addition
            temp2 = (S0 + maj) % 0x100000000
            
            # Update working variables
            h = g
            g = f
            f = e
            e = (d + temp1) % 0x100000000
            d = c
            c = b
            b = a
            a = (temp1 + temp2) % 0x100000000
        
        # Add compressed chunk to hash values with 32-bit mask
        h0 = (h0 + a) % 0x100000000
        h1 = (h1 + b) % 0x100000000
        h2 = (h2 + c) % 0x100000000
        h3 = (h3 + d) % 0x100000000
        h4 = (h4 + e) % 0x100000000
        h5 = (h5 + f) % 0x100000000
        h6 = (h6 + g) % 0x100000000
        h7 = (h7 + h) % 0x100000000
    
    # Produce the final hash value
    digest = ''
    for val in [h0, h1, h2, h3, h4, h5, h6, h7]:
        # Convert each 32-bit value to 8 hex digits
        hex_val = ''
        for j in range(8):
            # Extract 4 bits at a time
            shift_amt = 28 - (j * 4)
            nibble = mult_right_shift(val, shift_amt) & 0xF
            
            # Convert to hex digit using multiplication-based lookup
            hex_digits = "0123456789abcdef"
            hex_val += hex_digits[nibble]
            
        digest += hex_val
    
    return digest


def extreme_simple_pow(data, difficulty):
    """
    Extreme Multiplication-based Proof of Work implementation
    
    Args:
        data: Base data string to find proof of work for
        difficulty: Number of leading zeros required in hash
        
    Returns:
        tuple: (nonce, hash, attempts, time_taken)
    """
    # Target is a string with 'difficulty' number of leading zeros
    target = '0' * difficulty  
    nonce = 0
    start_time = time.time()
    attempts = 0
    
    while True:
        attempts += 1
        # Concatenate using multiplication (string duplication)
        test_data = data + str(nonce)
        hash_result = extreme_mult_sha256(test_data)
        
        # Check if hash starts with the target number of zeros
        match = True
        for i in range(difficulty):
            if hash_result[i] != '0':
                match = False
                break
                
        if match:
            time_taken = time.time() - start_time
            return nonce, hash_result, attempts, time_taken
        
        nonce += 1

if __name__ == "__main__":
    # Test the extreme multiplication-based implementation
    print("Testing extreme multiplication-based SHA-256 implementation...")
    
    # Very short test message to avoid excessive runtime
    test_message = "Hi"
    result = extreme_mult_sha256(test_message)
    print(f"Extreme Multiplication SHA-256 of '{test_message}': {result}")
    
