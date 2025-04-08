import requests
import json
import time
import hashlib
import random
import multiprocessing
from multiprocessing import Process, Queue # Value, Array not used in this version
# import ctypes # Not used in this version
import os
import sys
import subprocess # For attempting dependency installation

# --- Helper Class to Fetch Blockchain Info ---

class BitcoinBlockchainFetcher:
    """
    Fetches recent Bitcoin blockchain information from public APIs
    """
    def __init__(self, api_providers=None):
        """
        Initialize with multiple API providers for redundancy

        Args:
            api_providers (list, optional): List of API endpoint configurations
        """
        if api_providers is None:
            self.api_providers = [
                # BlockCypher API (often reliable)
                {
                    'url': 'https://api.blockcypher.com/v1/btc/main',
                    'method': 'get_blockcypher_info'
                },
                # Blockchain.info API (alternative)
                {
                    'url': 'https://blockchain.info/latestblock',
                    'method': 'get_blockchain_info'
                }
                # Add more potential API providers here if needed
            ]
        else:
            self.api_providers = api_providers

    def _parse_blockchain_info(self, data):
        """ Parses response from Blockchain.info API. """
        try:
            return {
                'block_height': data.get('height', 0),
                # Blockchain.info uses 'hash' for the current block, 'prev_block' for the previous hash
                # For mining, we need the hash of the *latest* block as the 'prev_hash' for the *next* block.
                # However, the simplest approach for simulation is often to use the latest block's data directly.
                # Let's assume 'prev_block' is what we need (hash of the block before the latest one).
                # If mining on top of 'latest', its hash becomes the 'prev_hash'. Let's use 'hash'.
                'prev_hash': data.get('hash', '0' * 64), # Using latest block hash as prev_hash for next
                'merkle_root': data.get('mrkl_root', '0' * 64),
                'timestamp': int(data.get('time', time.time())),
                'bits': data.get('bits', 0) # Keep as int for now, convert later if needed
            }
        except Exception as e:
            print(f"Error parsing Blockchain.info response: {e}")
            return None

    def _parse_blockcypher_info(self, data):
        """ Parses response from BlockCypher API. """
        try:
            # dateutil is needed for BlockCypher's ISO timestamp
            import dateutil.parser

            # BlockCypher gives info on the latest block
            # Its hash becomes the 'prev_hash' for the block we'd mine next
            timestamp_str = data.get('time')
            if timestamp_str:
                 # Ensure timestamp is timezone-aware before converting
                dt = dateutil.parser.isoparse(timestamp_str)
                timestamp = int(dt.timestamp())
            else:
                timestamp = int(time.time())

            return {
                'block_height': data.get('height', 0),
                'prev_hash': data.get('hash', '0' * 64), # Using latest block hash as prev_hash for next
                'merkle_root': data.get('merkle_root', '0' * 64),
                'timestamp': timestamp,
                'bits': data.get('bits', 0) # Keep as int for now, convert later if needed
            }
        except ImportError:
             print("Error: 'python-dateutil' package needed for BlockCypher timestamp parsing.")
             print("Please install it: pip install python-dateutil")
             return None
        except Exception as e:
            print(f"Error parsing BlockCypher response: {e}")
            return None

    def fetch_latest_block_info(self):
        """
        Fetch the latest Bitcoin block information, trying multiple providers.

        Returns:
            dict: Block information or default values if all APIs fail.
        """
        latest_info = None
        for provider in self.api_providers:
            print(f"Attempting to fetch data from: {provider['url']}")
            try:
                response = requests.get(provider['url'], timeout=10)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                data = response.json()

                if provider['method'] == 'get_blockchain_info':
                    latest_info = self._parse_blockchain_info(data)
                elif provider['method'] == 'get_blockcypher_info':
                    latest_info = self._parse_blockcypher_info(data)
                # Add more methods for other providers here

                if latest_info:
                    print(f"Successfully fetched data from {provider['url']}")
                    # Convert bits to hex string here for consistency
                    if 'bits' in latest_info and isinstance(latest_info['bits'], int):
                         latest_info['bits'] = hex(latest_info['bits'])[2:] # Format as hex string '1d00ffff'
                    return latest_info

            except requests.exceptions.RequestException as e:
                print(f"Error fetching from {provider['url']}: {e}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {provider['url']}: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred with {provider['url']}: {e}")


        # If all providers fail, return a default block info
        print("Warning: Unable to fetch live block information from any provider. Using default values.")
        return {
            'block_height': 0,
            'prev_hash': '0' * 64,
            'merkle_root': '0' * 64,
            'timestamp': int(time.time()),
            'bits': '1d00ffff'  # Default easy difficulty target
        }

# --- Core Mining Logic ---

class BlockchainSolver:
    @staticmethod
    def mine_chunk(prefix, start_nonce, chunk_size, target_zeros, result_queue, process_id):
        """
        Mine a chunk of nonce values for a target number of leading zeros (simplified difficulty).

        Args:
            prefix (str): Blockchain data prefix (simplified representation).
            start_nonce (int): Starting nonce for this chunk.
            chunk_size (int): Number of nonces to try in this chunk.
            target_zeros (int): Number of leading hexadecimal zeros required.
            result_queue (Queue): Queue to report results back to the main process.
            process_id (int): Identifier for the process reporting.
        """
        current_nonce = start_nonce
        best_zeros_found_in_chunk = 0
        best_hash_in_chunk = None
        best_nonce_in_chunk = start_nonce # Track the nonce corresponding to the best hash

        print(f"[Process {process_id}] Starting chunk from nonce {start_nonce}...")

        for i in range(chunk_size):
            nonce_to_try = current_nonce
            # Combine blockchain data prefix with current nonce value
            guess_string = f"{prefix}{nonce_to_try}"

            # Apply double SHA-256 as used in Bitcoin
            first_hash = hashlib.sha256(guess_string.encode('utf-8')).digest()
            current_hash_bytes = hashlib.sha256(first_hash).digest()
            current_hash_hex = current_hash_bytes.hex() # Get hex representation

            # Count leading zeros (simplified difficulty check)
            zeros = 0
            for char in current_hash_hex:
                if char == '0':
                    zeros += 1
                else:
                    break

            # Check if it meets the target difficulty
            if zeros >= target_zeros:
                print(f"[Process {process_id}] Solution FOUND!")
                result_queue.put({
                    'found': True,
                    'nonce': nonce_to_try,
                    'hash': current_hash_hex,
                    'zeros': zeros,
                    'process_id': process_id
                })
                return # Solution found, exit this process's task

            # Track the best hash found within this chunk if target not met yet
            if zeros > best_zeros_found_in_chunk:
                best_zeros_found_in_chunk = zeros
                best_hash_in_chunk = current_hash_hex
                best_nonce_in_chunk = nonce_to_try

            # Increment nonce - Use sequential increment for more realistic simulation
            current_nonce = (current_nonce + 1) & 0xFFFFFFFF # Wrap around 32-bit unsigned integer range

            # Optional: Add a check to see if the result queue has a solution from another process
            # This can stop processes faster but adds overhead
            # if i % 10000 == 0: # Check every N iterations
            #    if not result_queue.empty():
            #        print(f"[Process {process_id}] Detected possible solution elsewhere, exiting chunk early.")
            #        return


        # If no solution meeting target_zeros was found in this chunk, report the best attempt
        print(f"[Process {process_id}] Chunk finished. Best zeros found in chunk: {best_zeros_found_in_chunk}")
        if best_hash_in_chunk:
            result_queue.put({
                'found': False, # Did not meet target_zeros
                'nonce': best_nonce_in_chunk, # Nonce for the best hash found
                'hash': best_hash_in_chunk,
                'zeros': best_zeros_found_in_chunk,
                'process_id': process_id
            })
        else:
             # Should not happen if chunk_size > 0, but handle defensively
             result_queue.put({
                'found': False,
                'zeros': 0,
                'process_id': process_id
            })


# --- Multi-Process Management ---

class MultiProcessBlockchainMiner:
    def __init__(self, num_processes=None):
        """
        Initialize multi-process blockchain miner.

        Args:
            num_processes (int, optional): Number of processes to use.
                                           Defaults to number of CPU cores.
        """
        if num_processes is None:
            try:
                num_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                print("Warning: Could not detect CPU count. Defaulting to 1 process.")
                num_processes = 1
        self.num_processes = num_processes
        print(f"Miner initialized to use {self.num_processes} processes.")

    def mine_block(self, blockchain_prefix, target_zeros, max_iterations_total=10_000_000):
        """
        Mine a block using multiple processes seeking a specific number of leading zeros.

        Args:
            blockchain_prefix (str): Simplified prefix data for the block.
            target_zeros (int): Number of leading hexadecimal zeros required.
            max_iterations_total (int): Maximum *total* nonces to try across all processes.

        Returns:
            dict: Mining result (nonce, hash, etc.) if found within iterations, else None or best attempt.
        """
        print(f"\n--- Starting multi-process mining ---")
        print(f"Target: {target_zeros} leading zeros")
        print(f"Max total nonces: {max_iterations_total:,}")
        print(f"Processes: {self.num_processes}")
        print(f"Prefix data: {blockchain_prefix}")

        result_queue = multiprocessing.Queue()
        processes = []

        # Calculate nonce chunk size for each process
        # Ensure chunk_size is at least 1 if max_iterations_total is small
        chunk_size_per_process = max(1, max_iterations_total // self.num_processes)
        actual_total_iterations = chunk_size_per_process * self.num_processes
        print(f"Nonces per process chunk: {chunk_size_per_process:,}")
        print(f"Actual total nonces to be checked: {actual_total_iterations:,}")


        start_time = time.time()
        overall_best_result = {'found': False, 'zeros': -1} # Track best result if target not met


        # Create and start processes
        for i in range(self.num_processes):
            # Assign a unique starting nonce for each process to avoid overlap
            # Use a large random offset multiplied by process index for better distribution
            start_nonce = random.randint(0, 0xFFFFFFFF) # Each process starts randomly
            # Alternative: start_nonce = (i * chunk_size_per_process) & 0xFFFFFFFF # Sequential chunks

            p = multiprocessing.Process(
                target=BlockchainSolver.mine_chunk,
                args=(blockchain_prefix, start_nonce, chunk_size_per_process, target_zeros, result_queue, i)
            )
            processes.append(p)
            p.start()
            print(f"Started process {i} with start_nonce {start_nonce}")

        # Wait for results
        solution_found = None
        results_received = 0
        while results_received < self.num_processes:
            try:
                # Wait for a result from any process
                result = result_queue.get(timeout=1) # Use timeout to allow periodic checks/updates
                results_received += 1
                print(f"Result received from process {result.get('process_id', 'N/A')}. "
                      f"Found: {result.get('found')}, Zeros: {result.get('zeros', 'N/A')}. "
                      f"({results_received}/{self.num_processes} processes finished)")


                if result['found']: # Check if this result meets the target
                    solution_found = result
                    print(f"\nSOLUTION FOUND by process {solution_found['process_id']}!")
                    break # Exit the loop as soon as a solution is found
                else:
                    # If target not met, track the best result seen so far
                     if result.get('zeros', -1) > overall_best_result['zeros']:
                         overall_best_result = result

            except multiprocessing.queues.Empty:
                 # Timeout occurred, check if processes are still alive
                 if not any(p.is_alive() for p in processes):
                     print("All processes seem to have finished or terminated unexpectedly.")
                     break
                 # Optional: Print a status update during long runs
                 # elapsed = time.time() - start_time
                 # print(f"Mining... {elapsed:.1f}s elapsed. {results_received}/{self.num_processes} chunks done.")
                 continue # Continue waiting if processes are alive and no result yet

        # Cleanup: Terminate any remaining processes forcefully if a solution was found
        if solution_found:
            print("Terminating remaining processes...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"Terminating process {i}...")
                    p.terminate() # Forcefully stop
                    p.join(timeout=1) # Wait briefly for termination
                    if p.is_alive():
                         print(f"Warning: Process {i} did not terminate gracefully.")
        else:
             print("No solution found meeting the target. Waiting for all processes to join naturally...")
             # Ensure all processes have finished cleanly if no solution was found
             for i, p in enumerate(processes):
                 p.join() # Wait for process to complete its chunk
                 print(f"Process {i} joined.")


        end_time = time.time()
        duration = end_time - start_time
        # Calculate hashrate based on the actual number of hashes performed
        total_hashes_calculated = actual_total_iterations if not solution_found else "N/A (stopped early)" # Approximat
        hashrate = (actual_total_iterations / duration) if duration > 0 else 0

        print("\n--- Mining Finished ---")
        if solution_found:
            print("===== BLOCK FOUND! =====")
            print(f"Nonce: {solution_found['nonce']}")
            print(f"Hash: {solution_found['hash']}")
            print(f"Leading zeros: {solution_found['zeros']} (Target: >= {target_zeros})")
            print(f"Found by Process: {solution_found['process_id']}")
        elif overall_best_result['zeros'] != -1:
             print("===== TARGET NOT REACHED =====")
             print("Best attempt:")
             print(f"Nonce: {overall_best_result['nonce']}")
             print(f"Hash: {overall_best_result['hash']}")
             print(f"Leading zeros: {overall_best_result['zeros']} (Target: >= {target_zeros})")
             print(f"Found by Process: {overall_best_result['process_id']}")
        else:
            print("===== TARGET NOT REACHED (No valid hashes reported) =====")


        print(f"\nMining stats:")
        print(f"Time spent: {duration:.2f} seconds")
        print(f"Total nonces checked: {total_hashes_calculated}")
        print(f"Approximate Hashrate: {hashrate:,.0f} H/s") # Hashes per second

        return solution_found if solution_found else overall_best_result # Return solution or best attempt

# --- Simulation Orchestrator ---

class BlockchainMiningSimulator:
    """
    Simulates the Bitcoin mining process using fetched data and multi-processing.
    Uses a simplified block data representation and difficulty check (leading zeros).
    """
    def __init__(self):
        self.fetcher = BitcoinBlockchainFetcher()
        self.miner = MultiProcessBlockchainMiner() # Uses cpu_count by default

    def prepare_mining_data_prefix(self, block_info):
        """
        Prepare the simplified blockchain data prefix string for mining simulation.
        NOTE: This is a highly simplified representation, not the actual binary header.

        Args:
            block_info (dict): Block information fetched from API.

        Returns:
            str: Prepared blockchain data prefix string.
        """
        # Extract necessary components, providing defaults
        # Use the hash of the latest block as the 'previous hash' for the block we are trying to mine
        prev_hash = block_info.get('prev_hash', '0' * 64)
        merkle_root = block_info.get('merkle_root', '0' * 64)
        timestamp = str(block_info.get('timestamp', int(time.time())))
        bits = str(block_info.get('bits', '1d00ffff')) # Already converted to hex string by fetcher

        # --- SIMPLIFICATION ---
        # Create a shortened string representation for the simulation prefix.
        # Real mining hashes the 80-byte binary header.
        prefix_prev_hash = prev_hash[:16] # Use more bytes for better simulation
        prefix_merkle_root = merkle_root[:16]
        # Combine components into the prefix string. Nonce will be appended later.
        blockchain_prefix = f"{prefix_prev_hash}{prefix_merkle_root}{timestamp}{bits}"
        # --------------------

        print("\n===== Preparing Mining Data (Simplified Prefix) =====")
        print(f"Using Previous Block Hash (truncated): {prefix_prev_hash}...")
        print(f"Using Merkle Root (truncated): {prefix_merkle_root}...")
        print(f"Using Timestamp: {timestamp}")
        print(f"Using Bits (difficulty): {bits}")
        print(f"Full Prefix String for Hashing: {blockchain_prefix}") # Show the actual prefix

        return blockchain_prefix

    def run_mining_simulation(self, difficulties_to_try=None, iterations_per_difficulty=None):
        """
        Run the mining simulation loop for different simplified difficulties.

        Args:
            difficulties_to_try (list, optional): List of target leading zeros to attempt.
                                                  Defaults to [2, 3, 4, 5].
            iterations_per_difficulty (int, optional): Base number of iterations for the easiest difficulty.
                                                       Adjusted automatically for harder difficulties.
                                                       Defaults to 1,000,000.
        """
        print("Starting Bitcoin Block Mining Simulation...")
        print("Fetching latest block information...")

        block_info = self.fetcher.fetch_latest_block_info()

        print("\n===== Fetched Block Information =====")
        if block_info:
             # Print fetched info nicely
            print(f"  Block Height: {block_info.get('block_height', 'N/A')}")
            print(f"  Prev Block Hash: {block_info.get('prev_hash', 'N/A')}")
            print(f"  Merkle Root: {block_info.get('merkle_root', 'N/A')}")
            print(f"  Timestamp: {block_info.get('timestamp', 'N/A')}")
            print(f"  Bits (Difficulty): {block_info.get('bits', 'N/A')}")

            blockchain_prefix = self.prepare_mining_data_prefix(block_info)

            iterations_per_difficulty = 1_000_000_000 # Base iterations for easiest target
            target_zeros = 7

            while True:
                 # Rough adjustment of iterations based on difficulty (each extra zero is ~16x harder)
                 # This scaling factor is approximate for leading hex zeros
                 scaling_factor = 16 ** (target_zeros)
                 max_total_iterations = int(iterations_per_difficulty * scaling_factor)

                 # Call the multi-process miner
                 result = self.miner.mine_block(
                     blockchain_prefix,
                     target_zeros,
                     max_total_iterations
                 )

                 # Optional: Add a pause or check if user wants to continue
                 # input(f"\nPress Enter to attempt next difficulty level (or Ctrl+C to stop)...")

        else:
            print("Could not run simulation as block information fetch failed.")

# --- Main Execution ---

def main():
    """
    Main execution function for the Bitcoin mining simulation.
    """
    print("======================================")
    print(" Bitcoin Block Mining Simulator (CPU)")
    print("======================================")
    print("Note: This script simulates the hashing process but does not interact")
    print("      with the live Bitcoin network or generate real rewards.")
    print("      It uses simplified data structures and difficulty checks.")
    print("-" * 38)

    # --- Dependency Check ---
    # Check for requests and python-dateutil, attempt install if missing
    try:
        import requests
        # Check for dateutil specifically, as it's conditionally imported later
        import dateutil.parser
        print("Required libraries (requests, python-dateutil) are installed.")
    except ImportError:
        print("One or more required libraries ('requests', 'python-dateutil') not found.")
        try:
            print("Attempting to install missing libraries using pip...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', 'python-dateutil'])
            print("Libraries installed successfully. Please re-run the script.")
            # Exit after attempting install, user needs to restart
            sys.exit(0)
        except Exception as e:
            print(f"Error: Failed to install libraries automatically: {e}")
            print("Please install them manually:")
            print("  pip install requests python-dateutil")
            sys.exit(1) # Exit if dependencies cannot be installed

    # --- Run Simulation ---
    simulator = BlockchainMiningSimulator()
    # You can customize the difficulties and base iterations here:
    # e.g., simulator.run_mining_simulation(difficulties_to_try=[4, 5], iterations_per_difficulty=5000000)
    simulator.run_mining_simulation()

    print("\nSimulation finished.")


if __name__ == "__main__":
    # Crucial for Windows compatibility when using multiprocessing
    multiprocessing.freeze_support()
    main()
