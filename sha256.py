import hashlib
import time
import random
import multiprocessing
from multiprocessing import Process, Queue, Value, Array
import ctypes

class BlockchainSolver:
    @staticmethod
    def mine_chunk(prefix, start_nonce, chunk_size, target_zeros, result_queue):
        """
        Mine a chunk of nonce values
        
        Args:
            prefix (str): Blockchain data prefix
            start_nonce (int): Starting nonce for this chunk
            chunk_size (int): Number of nonces to try
            target_zeros (int): Number of leading zeros required
            result_queue (Queue): Queue to report results
        """
        current_guess = start_nonce
        best_zeros = 0
        best_hash = None
        
        for _ in range(chunk_size):
            # Combine blockchain data with current nonce value
            guess_string = f"{prefix}{current_guess}"
            
            # Apply double SHA-256 as used in Bitcoin
            first_hash = hashlib.sha256(guess_string.encode()).digest()
            current_hash = hashlib.sha256(first_hash).digest()
            current_hash_hex = current_hash.hex()
            
            # Count leading zeros
            zeros = 0
            for char in current_hash_hex:
                if char == '0':
                    zeros += 1
                else:
                    break
            
            # Check if better than current best
            if zeros > best_zeros:
                best_zeros = zeros
                best_hash = current_hash_hex
                
                # If target difficulty reached, report success
                if zeros >= target_zeros:
                    result_queue.put({
                        'found': True,
                        'nonce': current_guess,
                        'hash': current_hash_hex,
                        'zeros': zeros
                    })
                    return
            
            # Increment nonce with some randomness
            current_guess = (current_guess + random.randint(1, 1000)) & 0xFFFFFFFF
        
        # Report best attempt if no solution found
        if best_hash:
            result_queue.put({
                'found': False,
                'nonce': current_guess,
                'hash': best_hash,
                'zeros': best_zeros
            })

class MultiProcessBlockchainMiner:
    def __init__(self, num_processes=None):
        """
        Initialize multi-process blockchain miner
        
        Args:
            num_processes (int, optional): Number of processes to use. 
                                           Defaults to number of CPU cores.
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        self.num_processes = num_processes
    
    def mine_block(self, blockchain_data, target_zeros, max_iterations=100_000_000):
        """
        Mine a block using multiple processes
        
        Args:
            blockchain_data (str): Prefix data for the block
            target_zeros (int): Number of leading zeros required
            max_iterations (int): Maximum total iterations to try
        
        Returns:
            dict: Mining result with nonce, hash, etc. if found
        """
        print(f"Starting multi-process mining:")
        print(f"Processes: {self.num_processes}")
        print(f"Target zeros: {target_zeros}")
        print(f"Max total iterations: {max_iterations:,}")
        
        # Prepare multiprocessing components
        result_queue = multiprocessing.Queue()
        
        # Calculate chunk size for each process
        chunk_size = max_iterations // self.num_processes
        
        # Start timing
        start_time = time.time()
        
        # Create and start processes
        processes = []
        for i in range(self.num_processes):
            # Use different starting points for each process
            start_nonce = random.randint(0, 2**32)
            p = multiprocessing.Process(
                target=BlockchainSolver.mine_chunk,
                args=(blockchain_data, start_nonce, chunk_size, target_zeros, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Wait for a solution or all processes to complete
        solution = None
        while not solution:
            try:
                # Check for results with a timeout
                result = result_queue.get(timeout=0.1)
                
                # If a solution is found
                if result['found']:
                    solution = result
                    # Terminate all other processes
                    for p in processes:
                        p.terminate()
                    break
            except:
                # Check if all processes are done
                if not any(p.is_alive() for p in processes):
                    break
        
        # Ensure all processes are terminated
        for p in processes:
            p.terminate()
            p.join()
        
        # Calculate mining statistics
        duration = time.time() - start_time
        total_hashes = chunk_size * self.num_processes
        hashrate = total_hashes / duration
        
        # Prepare and print results
        if solution:
            print("\n===== BLOCK FOUND! =====")
            print(f"Nonce: {solution['nonce']}")
            print(f"Hash: {solution['hash']}")
            print(f"Leading zeros: {solution['zeros']}")
        else:
            print("\n===== NO BLOCK FOUND =====")
        
        print(f"\nMining stats:")
        print(f"Time spent: {duration:.2f} seconds")
        print(f"Total hashes: {total_hashes:,}")
        print(f"Hashrate: {hashrate:.0f} H/s")
        
        return solution

# Example usage
def main():
    # Create a simplified blockchain header
    prev_hash = "000000000000000000152678f83ec36cf273883b887d1c5110f33e4a4add9022"
    merkle_root = "e62211ec1ff6d3f0111880ac06c2436dd3d47b71630a19bb7a75e62e6ffce28b"
    timestamp = int(time.time())
    bits = "1d00ffff"
    
    # Create shortened version for demo (real blockchain would use full data)
    blockchain_data = f"{prev_hash[:8]}{merkle_root[:8]}{timestamp}{bits}"
    
    print("Multi-Process Bitcoin-style Blockchain Mining")
    print("===========================================")
    print(f"Blockchain data: {blockchain_data}")
    
    # Create multi-process miner
    miner = MultiProcessBlockchainMiner()
    
    # Try different difficulty levels
    for difficulty in range(2, 9):
        print(f"\nMining with difficulty {difficulty} (seeking {difficulty} leading zeros)")
        
        # Adjust iterations based on difficulty
        iterations = 100_000_000 * (4 ** (difficulty - 2))
        
        # Start mining
        result = miner.mine_block(blockchain_data, difficulty, iterations)

if __name__ == "__main__":
    # Needed for Windows multiprocessing support
    multiprocessing.freeze_support()
    main()