import os
import sys
import uuid
from datetime import datetime
import subprocess
import networkx as nx
from itertools import combinations

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.to_var import process_binary_string
from utils.append import update_cnf_file

import argparse
import concurrent.futures
import psutil  # Add this import

def graph6_to_upper_triangle_subgraphs(g6, m):
    """
    Convert a graph6 string to a list of upper triangle adjacency matrix strings
    for all non-isomorphic subgraphs with m vertices removed.

    Parameters:
        g6 (str): The graph6 string representation of the graph.
        m (int): Number of vertices to remove.

    Returns:
        list: List of binary strings representing the upper triangle of the adjacency matrix
              for each non-isomorphic subgraph.
    """
    G = nx.from_graph6_bytes(g6.encode())
    print(f"Processing graph with {G.number_of_nodes()} nodes")  # Add debug print
    order = G.number_of_nodes() - m  # Order of subgraphs
    
    # Generate non-isomorphic subgraphs
    subgraphs = []
    seen_canonical_forms = set()
    
    for vertices in combinations(G.nodes, order):
        subgraph = G.subgraph(vertices).copy()
        
        # Generate a canonical form for the subgraph
        canonical_form = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(subgraph)
        
        if canonical_form not in seen_canonical_forms:
            adj_matrix = nx.to_numpy_array(subgraph, dtype=int)
            upper_triangle_string = ''.join(
                str(adj_matrix[i, j])
                for j in range(order)
                for i in range(j)
            )
            print(f"Subgraph found: {upper_triangle_string}")  # Add debug print
            subgraphs.append(upper_triangle_string)
            seen_canonical_forms.add(canonical_form)
    
    print(f"Found {len(subgraphs)} non-isomorphic subgraphs:")
    for subgraph in subgraphs:
        print(subgraph)
    
    return subgraphs

def canonize_subgraph(subgraph_binary):
    print(f"Canonizing subgraph: {subgraph_binary}")  # Add debug print
    result = subprocess.run(
        ['./nauty2_8_8/RCL', subgraph_binary],
        capture_output=True, text=True
    )
    
    # Print full output for debugging
    print("RCL output:")
    print(result.stdout)
    
    # Extract the output string
    output_lines = result.stdout.splitlines()
    for line in output_lines:
        if line.startswith("Output string:"):
            return line.split(":")[1].strip()
    
    return ""

def process_subgraphs(subgraphs):
    """
    Convert each 01 string in subgraphs to variables using process_binary_string,
    but only for the actual edges in the subgraph.

    Parameters:
        subgraphs (list): List of binary strings representing subgraphs.

    Returns:
        list: List of variable strings (positive for 1, negative for 0).
    """
    variable_strings = []
    for subgraph in subgraphs:
        print(f"\nProcessing subgraph: {subgraph}")  # Add debug print
        canonized_binary = canonize_subgraph(subgraph)
        print(f"Canonized binary: {canonized_binary}")  # Add debug print
        
        variables = []
        for i, bit in enumerate(canonized_binary):
            if bit == '1':
                variables.append(str(i + 1))
            elif bit == '0':
                variables.append(str(-(i + 1)))
        
        print(f"Variables: {variables}")  # Add debug print
        variable_strings.append(' '.join(variables))
    return variable_strings

def add_unit_clauses_to_cnf(cnf_file, variable_strings, output_cnf_file):
    """
    Add variable strings as unit clauses to a CNF file.

    Parameters:
        cnf_file (str): Path to the input CNF file.
        variable_strings (list): List of variable strings (e.g., "1 2 -3 4 -5").
        output_cnf_file (str): Path to the output CNF file.
    """
    for variables in variable_strings:
        # Convert variable string to list of integers
        unit_clauses = [int(var) for var in variables.split()]
        
        # Add unit clauses to the CNF file
        update_cnf_file(cnf_file, unit_clauses, output_cnf_file)
        print(f"Added unit clauses to {output_cnf_file}: {unit_clauses}")

def solve_cnf(output_cnf_file, order):
    # Print process information
    print(f"Starting solver for {output_cnf_file}")
    
    solver_command = [
        './cadical-rcl/build/cadical',
        output_cnf_file,
        '--order',
        str(order)
    ]
    
    # Use Popen instead of run to track the process
    process = subprocess.Popen(solver_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = process.pid
    print(f"Solver process started with PID: {pid}")
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    
    # Create log file
    log_file = output_cnf_file + '.log'
    with open(log_file, 'w') as f:
        f.write("Solver command: " + ' '.join(solver_command) + '\n\n')
        f.write("Solver output:\n")
        f.write(stdout.decode())
        if stderr:
            f.write("\nSolver errors:\n")
            f.write(stderr.decode())
    
    print(f"Solver process {pid} completed")
    return log_file, pid

#graph6 of conway-31 = "^tq?GdA?gK?_?@CH__aAGCO??GG?COOOG?`CG??G?_CG@ARCAO@?CG?`???W?@?gGA_???G??GA?_@G"
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process graph6 strings and generate CNF files.')
    parser.add_argument('--graph6', type=str, default="^tq?GdA?gK?_?@CH__aAGCO??GG?COOOG?`CG??G?_CG@ARCAO@?CG?`???W?@?gGA_???G??GA?_@G",
                       help='graph6 string representation of the graph (default: Conway-31)')
    parser.add_argument('--remove_order', type=int, default=0,
                       help='number of vertices to remove (default: 0)')
    parser.add_argument('--cnf_file', type=str, default="constraints_31_0_1",
                       help='path to the input CNF file (default: constraints_31_0_1)')
    parser.add_argument('--order', type=int, default=31,
                       help='order of the output graph (default: 31)')
    
    args = parser.parse_args()
    
    # Use arguments instead of hardcoded values
    graph6 = args.graph6
    m = args.remove_order
    cnf_file = args.cnf_file
    order = args.order
    
    subgraphs = graph6_to_upper_triangle_subgraphs(graph6, m)
    variable_strings = process_subgraphs(subgraphs)

    # Print the variable strings
    print("\nVariable representations:")
    for variables in variable_strings:
        print(variables)

    # Create a unique folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    folder_name = f"output_{timestamp}_{unique_id}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"\nCreated output folder: {folder_name}")
    print(f"Absolute path: {os.path.abspath(folder_name)}")  # Add debug print

    # Add unit clauses to CNF files
    for i, variables in enumerate(variable_strings):
        output_cnf_file = os.path.join(folder_name, f"subgraph_{i}.cnf")
        print(f"\nCreating CNF file: {output_cnf_file}")  # Add debug print
        print(f"Variables to write: {variables}")  # Add debug print
        add_unit_clauses_to_cnf(cnf_file, [variables], output_cnf_file)
        print(f"CNF file created: {output_cnf_file}")
        
        # Verify file exists
        if os.path.exists(output_cnf_file):
            print(f"File verification: {output_cnf_file} exists")
        else:
            print(f"ERROR: {output_cnf_file} was not created")

    # Call solver for each generated CNF file in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Track all solver PIDs
        solver_pids = set()
        
        futures = [
            executor.submit(solve_cnf, os.path.join(folder_name, f"subgraph_{i}.cnf"), order)
            for i in range(len(variable_strings))
        ]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                log_file, pid = future.result()
                solver_pids.add(pid)
                print(f"[{i+1}/{len(futures)}] Solver output saved to: {log_file}")
                print(f"Current active solver processes: {len(solver_pids)}")
            except Exception as e:
                print(f"[{i+1}/{len(futures)}] Error solving CNF file: {e}")
        
        print(f"\nTotal solver processes spawned: {len(solver_pids)}")