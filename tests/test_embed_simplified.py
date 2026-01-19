#!/usr/bin/python
from z3 import *
import sys
import re
import networkx as nx

"""
Simplified embeddability testing script.
Usage: python3 test_embed_simplified.py <solution_file> <order>

This script tests graph embeddability with:
- Predefined vectors for vertices 0-24
- Simple orthogonality constraints only for actual edges
- No complex assignment search or cross-product constraints
"""

def dot(v, w):
    """Dot product of two 3D vectors"""
    return (v[0]*w[0] + v[1]*w[1] + v[2]*w[2])

def not_zero(a):
    """Constraint that vector a is not the zero vector"""
    return Or(a[0] != 0, a[1] != 0, a[2] != 0)

def test_embeddability_simplified(graph_dict, order, g_sat, output_unsat_f="unsat_solutions.txt", output_sat_f="sat_solutions.txt", verify=False):
    """
    Simplified embeddability test using predefined vectors for first 25 vertices
    and only edge-based orthogonality constraints for the rest.
    
    Args:
        graph_dict: Dictionary of vertex -> list of neighbors
        order: Number of vertices
        g_sat: Graph identifier for logging
        output_unsat_f: File to log non-embeddable graphs
        output_sat_f: File to log embeddable graphs
        verify: Whether to verify and print solution
    """
    
    # Predefined vectors for the first 25 vertices (matching original script)
    predefined_vectors = [
        [-2, 1, 1],   # v0
        [1, -2, 1],   # v1
        [-1, 2, 1],   # v2
        [2, -1, 1],   # v3
        [-1, 1, 2],   # v4
        [2, 1, -1],   # v5
        [1, 2, 1],    # v6
        [1, -1, 2],   # v7
        [1, 2, -1],   # v8
        [2, 1, 1],    # v9
        [1, 1, -2],   # v10
        [1, 1, 2],    # v11
        [1, 0, 0],    # v12
        [0, 1, 0],    # v13
        [1, -1, 0],   # v14
        [-1, 1, 1],   # v15
        [1, -1, 1],   # v16
        [0, 0, 1],    # v17
        [1, 1, -1],   # v18
        [1, 0, -1],   # v19
        [0, 1, -1],   # v20
        [0, 1, 1],    # v21
        [1, 1, 1],    # v22
        [1, 0, 1],    # v23
        [1, 1, 0]     # v24
    ]
    
    print(f"Testing embeddability for graph with {order} vertices")
    
    s = Solver()
    ver = {}
    
    # Create variables for all vertices
    for i in range(order):
        ver[i] = (Real(f"ver{i}c1"), Real(f"ver{i}c2"), Real(f"ver{i}c3"))
        
        # Assign predefined vectors to first 25 vertices
        if i < 25 and i < len(predefined_vectors):
            print(f"Assigning predefined vector {predefined_vectors[i]} to vertex {i}")
            s.add(ver[i][0] == predefined_vectors[i][0])
            s.add(ver[i][1] == predefined_vectors[i][1])
            s.add(ver[i][2] == predefined_vectors[i][2])
        else:
            # For vertices 25+, ensure they're not zero vectors
            s.add(not_zero(ver[i]))
    
    # Add orthogonality constraints for each edge in the graph
    edge_count = 0
    for v in graph_dict:
        for w in graph_dict[v]:
            if v < w:  # Avoid duplicate constraints for undirected edges
                s.add(dot(ver[v], ver[w]) == 0)
                edge_count += 1
                if edge_count % 50 == 0:  # Progress indicator
                    print(f"Added {edge_count} edge constraints...")
    
    print(f"Total edge constraints added: {edge_count}")
    
    # Set timeout and solve
    s.set("timeout", 100000)  # 100 second timeout
    print("Solving...")
    result = s.check()
    
    if result == unknown:
        print("Timeout reached: Embeddability unknown")
        return False
    elif result == unsat:
        print("Not embeddable")
        with open(output_unsat_f, "a+") as f:
            f.write(g_sat + "\n")
        return True
    elif result == sat:
        print("Embeddable!")
        with open(output_sat_f, "a+") as f:
            f.write(g_sat + "\n")
        
        if verify:
            m = s.model()
            print("\nSolution found! Vector assignments:")
            for i in range(min(order, 30)):  # Show first 30 vertices
                if i < 25:
                    print(f"Vertex {i}: {predefined_vectors[i]} (predefined)")
                else:
                    try:
                        x_val = m.evaluate(ver[i][0], model_completion=True)
                        y_val = m.evaluate(ver[i][1], model_completion=True)
                        z_val = m.evaluate(ver[i][2], model_completion=True)
                        print(f"Vertex {i}: [{x_val}, {y_val}, {z_val}] (computed)")
                    except:
                        print(f"Vertex {i}: [unable to evaluate] (computed)")
            
            if order > 30:
                print(f"... and {order - 30} more vertices")
            
            # Verify orthogonality constraints for a sample of edges
            print("\nVerifying orthogonality for sample edges:")
            verified_edges = 0
            for v in graph_dict:
                for w in graph_dict[v]:
                    if v < w and verified_edges < 10:  # Check first 10 edges
                        try:
                            dot_result = m.evaluate(dot(ver[v], ver[w]), model_completion=True)
                            print(f"Edge ({v},{w}): dot product = {dot_result}")
                            verified_edges += 1
                        except:
                            print(f"Edge ({v},{w}): unable to verify")
            
        return True

def binary_to_edges(binary_str, order=None):
    """Convert binary string to edge list"""
    str_len = len(binary_str)
    if order is None:
        n = int((1 + (1 + 8*str_len)**0.5)/2)
        if n*(n-1)//2 != str_len:
            raise ValueError(f"Binary string length {str_len} is not a valid triangular number")
        order = n
    
    edge_lst = []
    idx = 0
    for j in range(order):
        for i in range(j):
            if binary_str[idx] == '1':
                edge_lst.append((i, j))
            idx += 1
    return edge_lst

def process_single_solution(binary_str, order, solution_num):
    """Process a single solution binary string"""
    print(f"\n=== Processing Solution {solution_num} ===")
    print(f"Binary string: {binary_str}")
    print(f"String length: {len(binary_str)}")
    
    try:
        # Convert binary string to edges
        edge_lst = binary_to_edges(binary_str, order)
        print(f"Number of edges: {len(edge_lst)}")
        
        # Create graph dictionary
        graph_dict = {i: [] for i in range(order)}
        for v, w in edge_lst:
            graph_dict[v].append(w)
            graph_dict[w].append(v)
        
        # Test embeddability
        print("Testing embeddability...")
        result = test_embeddability_simplified(
            graph_dict, 
            order, 
            binary_str, 
            verify=True
        )
        
        if result:
            print(f"Solution {solution_num}: Processing completed")
        else:
            print(f"Solution {solution_num}: Processing failed")
            
    except Exception as e:
        print(f"Error processing solution {solution_num}: {e}")
        import traceback
        traceback.print_exc()

def extract_solutions_from_file(filename):
    """Extract all binary strings that follow 'Solution: ' in the file"""
    solutions = []
    solution_pattern = re.compile(r'Solution[^:]*:\s*([01]+)')
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                match = solution_pattern.search(line)
                if match:
                    binary_str = match.group(1)
                    solutions.append((len(solutions) + 1, binary_str))
                    print(f"Found solution on line {line_num}: {binary_str}")
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return []
    
    return solutions

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 test_embed_simplified.py <solution_file> <order>")
        print("Example: python3 test_embed_simplified.py sic25_32.txt 25")
        sys.exit(1)
    
    filename = sys.argv[1]
    try:
        order = int(sys.argv[2])
    except ValueError:
        print("Error: Order must be an integer")
        sys.exit(1)
    
    print(f"Processing solutions from '{filename}' with order {order}")
    print("Using simplified embeddability testing (edge constraints only)")
    
    # Extract all solutions from the file
    solutions = extract_solutions_from_file(filename)
    
    if not solutions:
        print("No solutions found in the file")
        sys.exit(1)
    
    print(f"Found {len(solutions)} solution(s) to process")
    
    # Process each solution
    for solution_num, binary_str in solutions:
        process_single_solution(binary_str, order, solution_num)
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed {len(solutions)} solution(s)")
    print("Check 'sat_solutions.txt' for embeddable graphs")
    print("Check 'unsat_solutions.txt' for non-embeddable graphs")

if __name__ == "__main__":
    main() 