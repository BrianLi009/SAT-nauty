import numpy as np
import itertools
import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pysat.solvers import Cadical103
from pysat.formula import CNF

def colorable(adj_matrix_str):
    """
    Given a list of numbers representing the positions of 1s in the upper triangle
    of an adjacency matrix (column by column), return a valid coloring if possible.
    """
    # Convert string to list of integers by removing brackets and splitting by comma
    numbers = [int(x.strip()) for x in adj_matrix_str.strip('[]').split(',')]
    
    # Calculate order of graph (n)
    # For a n×n matrix, the upper triangle (excluding diagonal) has (n*(n-1))/2 elements
    # Find largest position in the input to determine matrix size
    max_pos = max(numbers)
    n = 1
    while (n * (n-1)) / 2 < max_pos:
        n += 1
    
    # Convert position list to adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)
    for pos in numbers:
        # Convert position to (i,j) coordinates in upper triangle
        # pos starts from 1, so subtract 1 to convert to 0-based indexing
        pos = pos - 1
        col = 0
        while pos >= col:
            pos -= col
            col += 1
        row = pos
        adj_matrix[row][col] = 1
        adj_matrix[col][row] = 1  # Mirror across diagonal
    
    # Convert adjacency matrix to edge list
    edge_lst = []
    vertices_lst = list(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] == 1:
                edge_lst.append((i, j))
    
    print (adj_matrix)
    
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
    return s.solve()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_010.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip()  # Remove trailing whitespace and newlines
                if line:  # Skip empty lines
                    result = colorable(line)
                    print(f"Line: {line}")
                    print(f"Result: {result}\n")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)