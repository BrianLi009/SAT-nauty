#!/usr/bin/env python3
"""
Enhanced script to convert multiple graph6 strings to canonical forms using RCL,
process a base CNF file, and organize all output into a specified folder.

Usage:
    python3 simple_graph6_to_canonical.py --graphs <graphs_file> --base_cnf <base_cnf_file> --output_dir <output_folder>
    python3 simple_graph6_to_canonical.py --interactive

Example:
    python3 simple_graph6_to_canonical.py --graphs graphs.txt --base_cnf base.cnf --output_dir results/
"""

import subprocess
import argparse
import sys
import os
import math
import shutil
from pathlib import Path

def decode_graph6_simple(graph6_string):
    """
    Simple graph6 decoder that doesn't require networkx.
    Returns the adjacency matrix as a list of lists.
    """
    try:
        # Remove any header if present
        if graph6_string.startswith('>>graph6<<'):
            graph6_string = graph6_string[10:]
        
        # Graph6 format: first character gives number of vertices
        if len(graph6_string) == 0:
            raise ValueError("Empty graph6 string")
        
        # For small graphs (≤62 vertices), first character is offset by 63
        if ord(graph6_string[0]) <= 126:
            n = ord(graph6_string[0]) - 63
            data_start = 1
        else:
            # For larger graphs, first 4 characters encode the size
            # This is a simplified version - may not handle all cases
            raise ValueError("Large graphs not supported in this simple decoder")
        
        if n <= 0:
            raise ValueError("Invalid graph size")
        
        # Calculate expected data length in characters (6 bits per character)
        bits_needed = n * (n - 1) // 2
        chars_needed = math.ceil(bits_needed / 6)
        
        if len(graph6_string) < data_start + chars_needed:
            raise ValueError(f"Graph6 string too short for {n} vertices (need {chars_needed} chars, got {len(graph6_string) - data_start})")
        
        # Initialize adjacency matrix
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Decode the bit string
        bit_string = ""
        for i in range(data_start, data_start + chars_needed):
            if i < len(graph6_string):
                c = ord(graph6_string[i]) - 63
                if c < 0 or c > 63:
                    raise ValueError(f"Invalid character in graph6 string: {graph6_string[i]}")
                # Convert to 6-bit binary
                bit_string += format(c, '06b')
        
        # Extract the relevant bits for the upper triangle
        upper_triangle_bits = bit_string[:bits_needed]
        
        # Fill the adjacency matrix
        bit_index = 0
        for j in range(n):
            for i in range(j):
                if bit_index < len(upper_triangle_bits):
                    adj_matrix[i][j] = int(upper_triangle_bits[bit_index])
                    adj_matrix[j][i] = int(upper_triangle_bits[bit_index])  # Mirror
                    bit_index += 1
        
        return adj_matrix
        
    except Exception as e:
        print(f"Error decoding graph6 string: {e}")
        return None

def adjacency_matrix_to_01_string(adj_matrix):
    """
    Convert adjacency matrix to 01 string representing upper triangle (column by column).
    """
    n = len(adj_matrix)
    upper_triangle = []
    
    for j in range(n):
        for i in range(j):
            upper_triangle.append(str(adj_matrix[i][j]))
    
    return ''.join(upper_triangle)

def call_rcl(bit_string, color_representation=None):
    """
    Call RCL to get the canonical form of the 01 string and vertex mapping.
    Returns (canonical_string, vertex_mapping) where vertex_mapping is a dict mapping original -> canonical.
    """
    # Check if RCL executable exists
    rcl_path = "./nauty2_8_8/RCL"
    rcl_color_path = "./nauty2_8_8/RCL-color"
    
    if color_representation and os.path.exists(rcl_color_path):
        cmd = [rcl_color_path, bit_string, color_representation]
        print(f"Calling RCL-color with color representation {color_representation}")
    elif os.path.exists(rcl_path):
        cmd = [rcl_path, bit_string]
        print("Calling RCL")
    else:
        print("Error: RCL executable not found. Please ensure nauty2_8_8 is built.")
        return None, None
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("RCL output:")
            print(stdout)
            
            # Extract the output string from RCL
            output_string = None
            if "Output string:" in stdout:
                output_string = stdout.split("Output string: ")[-1].strip()
            else:
                print("Warning: Could not find 'Output string:' in RCL output")
                return None, None
            
            # Extract vertex mapping from RCL output
            vertex_mapping = {}
            lines = stdout.split('\n')
            in_permutation_section = False
            
            for line in lines:
                if "Permutation (input -> canonical):" in line:
                    in_permutation_section = True
                    continue
                elif in_permutation_section:
                    if "->" in line and line.strip():
                        try:
                            # Parse lines like "0 -> 0"
                            parts = line.strip().split(" -> ")
                            if len(parts) == 2:
                                original_vertex = int(parts[0])
                                canonical_vertex = int(parts[1])
                                vertex_mapping[original_vertex] = canonical_vertex
                        except ValueError:
                            # Skip lines that don't match the expected format
                            continue
                    elif line.strip() == "":
                        # Empty line might indicate end of permutation section
                        continue
                    else:
                        # Non-empty line that doesn't match format, likely end of section
                        break
            
            return output_string, vertex_mapping
        else:
            print(f"Error running RCL (return code {process.returncode}):")
            print(stderr)
            return None, None
            
    except Exception as e:
        print(f"Failed to execute RCL: {str(e)}")
        return None, None

def add_vars_to_cnf(input_cnf_file, vars_list, output_cnf_file):
    """
    Add variables from vars_list as unit clauses to a CNF formula.
    Based on add_vars_to_cnf.py functionality.
    """
    try:
        # Read the original CNF file
        with open(input_cnf_file, 'r') as f:
            cnf_lines = f.readlines()
        
        # Parse header
        header = cnf_lines[0].strip().split()
        if len(header) < 4 or header[0] != 'p' or header[1] != 'cnf':
            raise ValueError("Invalid CNF header format")
        
        num_vars = int(header[2])
        num_clauses = int(header[3])
        
        # Add unit clauses
        new_clauses = [f"{var} 0\n" for var in vars_list]
        
        # Update header with new clause count
        new_header = f"{header[0]} {header[1]} {num_vars} {num_clauses + len(new_clauses)}\n"
        
        # Write to output file
        with open(output_cnf_file, 'w') as f:
            f.write(new_header)
            f.writelines(cnf_lines[1:])  # Write original clauses
            f.writelines(new_clauses)    # Write new unit clauses
        
        print(f"Successfully created CNF with {len(new_clauses)} additional unit clauses")
        return True
        
    except Exception as e:
        print(f"Error processing CNF file: {e}")
        return False

def process_graphs_batch(graphs_file, base_cnf_file, output_dir, color_representation=None):
    """
    Process multiple graphs from a file and create CNF files with canonical variables.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read all graph6 strings from the input file
    try:
        with open(graphs_file, 'r') as f:
            graph6_strings = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Graphs file '{graphs_file}' not found")
        return False
    
    print(f"Processing {len(graph6_strings)} graphs...")
    print("=" * 80)
    
    # Process each graph
    for i, graph6_string in enumerate(graph6_strings, 1):
        print(f"\nProcessing graph {i}/{len(graph6_strings)}: {graph6_string[:50]}...")
        print("-" * 60)
        
        try:
            # Convert graph6 to adjacency matrix
            adj_matrix = decode_graph6_simple(graph6_string)
            if adj_matrix is None:
                print(f"  Skipping graph {i} due to decoding error")
                continue
            
            n = len(adj_matrix)
            print(f"  Graph decoded: {n} nodes")
            
            # Convert to 01 string
            bit_string = adjacency_matrix_to_01_string(adj_matrix)
            print(f"  Original 01 string length: {len(bit_string)}")
            
            # Call RCL to get canonical form and vertex mapping
            result = call_rcl(bit_string, color_representation)
            if result is None or result[0] is None:
                print(f"  Skipping graph {i} due to RCL error")
                continue
            
            canonical_string, vertex_mapping = result
            
            print(f"  Canonical 01 string length: {len(canonical_string)}")
            
            # Convert canonical string to variable list
            variables = []
            for j, char in enumerate(canonical_string, start=1):
                if char == '1':
                    variables.append(j)
                else:
                    variables.append(-j)
            
            # Create output files for this graph
            graph_prefix = f"graph_{i:03d}"
            
            # Save canonical 01 string
            canonical_file = os.path.join(output_dir, f"{graph_prefix}_canonical.txt")
            with open(canonical_file, 'w') as f:
                f.write(canonical_string + '\n')
            
            # Save variables
            vars_file = os.path.join(output_dir, f"{graph_prefix}_vars.txt")
            with open(vars_file, 'w') as f:
                f.write(' '.join(map(str, variables)) + '\n')
            
            # Save vertex mapping
            mapping_file = os.path.join(output_dir, f"{graph_prefix}_mapping.txt")
            with open(mapping_file, 'w') as f:
                f.write(f"Vertex Mapping (Original -> Canonical)\n")
                f.write("=" * 40 + "\n")
                for original_vertex in sorted(vertex_mapping.keys()):
                    canonical_vertex = vertex_mapping[original_vertex]
                    f.write(f"Vertex {original_vertex} -> Vertex {canonical_vertex}\n")
            
            # Create CNF with variables added
            cnf_file = os.path.join(output_dir, f"{graph_prefix}_with_vars.cnf")
            if add_vars_to_cnf(base_cnf_file, variables, cnf_file):
                print(f"  Created CNF file: {cnf_file}")
            else:
                print(f"  Failed to create CNF file for graph {i}")
            
            # Save summary info
            summary_file = os.path.join(output_dir, f"{graph_prefix}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Graph {i} Summary\n")
                f.write("=" * 30 + "\n")
                f.write(f"Original graph6: {graph6_string}\n")
                f.write(f"Vertices: {n}\n")
                f.write(f"Original 01 string: {bit_string}\n")
                f.write(f"Canonical 01 string: {canonical_string}\n")
                f.write(f"Variables: {' '.join(map(str, variables))}\n")
                f.write(f"\nVertex Mapping (Original -> Canonical):\n")
                for original_vertex in sorted(vertex_mapping.keys()):
                    canonical_vertex = vertex_mapping[original_vertex]
                    f.write(f"  Vertex {original_vertex} -> Vertex {canonical_vertex}\n")
            
            print(f"  ✓ Graph {i} processed successfully")
            
        except Exception as e:
            print(f"  ✗ Error processing graph {i}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Batch processing complete. Results saved in: {output_dir}")
    
    # Create a summary report
    summary_report = os.path.join(output_dir, "batch_summary.txt")
    with open(summary_report, 'w') as f:
        f.write(f"Batch Processing Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total graphs processed: {len(graph6_strings)}\n")
        f.write(f"Base CNF file: {base_cnf_file}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Processing completed at: {os.popen('date').read().strip()}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Convert multiple graph6 strings to canonical forms using RCL, process base CNF, and organize output'
    )
    parser.add_argument('--graphs', '-g', help='File containing graph6 strings (one per line)')
    parser.add_argument('--base_cnf', '-c', help='Base CNF file to add variables to')
    parser.add_argument('--output_dir', '-o', help='Output directory for all results')
    parser.add_argument('--color', '-cl', help='Color representation for RCL-color')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        graphs_file = input("Enter path to graphs file: ").strip()
        base_cnf_file = input("Enter path to base CNF file: ").strip()
        output_dir = input("Enter output directory: ").strip()
        color_representation = input("Enter color representation (optional, press Enter to skip): ").strip()
        if not color_representation:
            color_representation = None
    else:
        # Command line mode
        if not args.graphs or not args.base_cnf or not args.output_dir:
            parser.print_help()
            print("\nError: --graphs, --base_cnf, and --output_dir are required (unless using --interactive)")
            sys.exit(1)
        
        graphs_file = args.graphs
        base_cnf_file = args.base_cnf
        output_dir = args.output_dir
        color_representation = args.color
    
    # Validate input files
    if not os.path.exists(graphs_file):
        print(f"Error: Graphs file '{graphs_file}' not found")
        sys.exit(1)
    
    if not os.path.exists(base_cnf_file):
        print(f"Error: Base CNF file '{base_cnf_file}' not found")
        sys.exit(1)
    
    # Process the graphs
    success = process_graphs_batch(graphs_file, base_cnf_file, output_dir, color_representation)
    
    if success:
        print("\n✅ All processing completed successfully!")
    else:
        print("\n❌ Processing completed with errors. Check the output for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
