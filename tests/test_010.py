import numpy as np
import itertools
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pysat.solvers import Cadical103
from pysat.formula import CNF

def colorable(binary_str):
    """
    Given a binary string representing the adjacency matrix, return a valid coloring if possible.
    """
    #print(f"\nDebug: Processing binary string: {binary_str}")
    #print(f"Debug: String length: {len(binary_str)}")
    
    # Calculate n from string length
    n = int((1 + np.sqrt(1 + 8 * len(binary_str))) / 2)
    #print(f"Debug: Calculated order (n): {n}")
    #print(f"Debug: Verification: n(n-1)/2 should equal string length: {n*(n-1)//2} == {len(binary_str)}")
    
    # Convert binary string to adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)
    idx = 0
    for j in range(n):  # Column
        for i in range(j):  # Row (up to j)
            #print(f"Debug: Processing bit {idx} ({i},{j}): {binary_str[idx]}")
            if binary_str[idx] == '1':
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # Mirror across diagonal
            idx += 1
    
    #print(f"Debug: Resulting adjacency matrix:")
    #print(adj_matrix)
    
    # Convert adjacency matrix to edge list
    edge_lst = []
    vertices_lst = list(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] == 1:
                edge_lst.append((i, j))
    
    # SAT solving part
    s = Cadical103()
    cnf = CNF()
    edge_lst = [(a+1, b+1) for (a,b) in edge_lst]
    
    for edge in edge_lst:
        cnf.append(tuple([-edge[0],-edge[1]]))  # no two adjacent vertices can be both 1
        
    potential_triangles = list(itertools.combinations(vertices_lst, 3))
    for triangle in potential_triangles:
        v1 = triangle[0] + 1  # +1 because we're using 1-based indexing for SAT
        v2 = triangle[1] + 1
        v3 = triangle[2] + 1
        if ((v1, v2) in edge_lst or (v2,v1) in edge_lst) and \
           ((v2, v3) in edge_lst or (v3,v2) in edge_lst) and \
           ((v1, v3) in edge_lst or (v3,v1) in edge_lst):
            cnf.append((v1,v2,v3))
            cnf.append((-v1, -v2))
            cnf.append((-v2, -v3))
            cnf.append((-v1, -v3))
    
    s.append_formula(cnf.clauses)
    is_sat = s.solve()
    
    if is_sat:
        # Get the model (variable assignments)
        model = s.get_model()
        return is_sat, model
    else:
        return is_sat, None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_010.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r') as file:
            for line in file:
                if line.startswith('Solution'):
                    # Extract the binary string after "Solution N: "
                    binary_str = line.split(': ')[1].strip()
                    is_sat, model = colorable(binary_str)
                    print(f"Solution: {binary_str}")
                    print(f"Result: {is_sat}")
                    
                    if is_sat:
                        print("Variable assignments:")
                        for var in model:
                            var_num = abs(var)
                            var_val = var > 0
                            print(f"Variable {var_num}: {var_val}")
                    print()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)