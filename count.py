import re

def count_multiplications(filename):
    # Read the file content
    with open(filename, 'r') as file:
        code_content = file.read()
    
    # Tracking multiplications
    mult_operations = []
    
    # Comprehensive regex patterns to catch various multiplication scenarios
    patterns = [
        # Direct numeric multiplications
        r'(\d+)\s*\*\s*(\d+)',
        
        # Multiplications in function calls and assignments
        r'result\s*=\s*(\d+)\s*\*\s*(\d+)',
        r'result\s*\*=\s*(\d+)',
        
        # Multiplications in for loops
        r'for\s*.*\s*in\s*range\(.*\):\s*.*\s*(\d+)\s*\*\s*(\d+)',
        
        # Multiplication in SHA-256 constants
        r'k\s*=\s*\[\s*([^]]+)\s*\]',  # Capture SHA-256 constants
        
        # Multiplication-based utility functions
        r'mult_pow\((\d+),\s*(\d+)\)',
        r'result\s*=\s*result\s*\*\s*base',
        r'result\s*\*\s*=\s*base',
        
        # Additional loop multiplication patterns
        r'.*:\s*.*\s*result\s*=\s*(\d+)\s*\*\s*(\d+)',
    ]
    
    # Combine patterns
    combined_pattern = '|'.join(patterns)
    
    # Find all matches
    matches = re.findall(combined_pattern, code_content, re.MULTILINE | re.DOTALL)
    
    # Process matches
    for match in matches:
        # Filter out empty matches and process
        numeric_matches = [m for m in match if m and m.strip()]
        
        for i in range(len(numeric_matches) - 1):
            try:
                left = int(numeric_matches[i])
                right = int(numeric_matches[i+1])
                
                result = left * right
                mult_operations.append({
                    'left': left,
                    'right': right,
                    'result': result
                })
            except (ValueError, IndexError):
                continue
    
    # Special handling for SHA-256 constants
    constant_matches = re.findall(r'0x[0-9a-fA-F]+', code_content)
    constant_multiplications = []
    
    for i in range(len(constant_matches) - 1):
        try:
            left = int(constant_matches[i], 16)
            right = int(constant_matches[i+1], 16)
            
            result = left * right
            constant_multiplications.append({
                'left': left,
                'right': right,
                'result': result
            })
        except ValueError:
            continue
    
    # Combine and deduplicate operations
    all_operations = mult_operations + constant_multiplications
    
    # Remove duplicates
    unique_operations = []
    seen = set()
    for op in all_operations:
        key = (op['left'], op['right'], op['result'])
        if key not in seen:
            unique_operations.append(op)
            seen.add(key)
    
    return unique_operations

# Path to the file
filename = 'anisotropic_crystal_lattice_crypto.py'

# Count multiplications
mult_operations = count_multiplications(filename)

# Calculate total operations and sum
total_operations = len(mult_operations)
decimal_sum = sum(op['result'] for op in mult_operations)

print(f"Total Multiplication Operations: {total_operations}")
print(f"Decimal Sum of Multiplications: {decimal_sum}")

# Print detailed multiplication locations
print("\nDetailed Multiplication Locations:")
for op in mult_operations:
    print(f"{op['left']} * {op['right']} = {op['result']}")