import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import os
import time
import shutil

# Vector coordinates
v1 = (1,0,0)
v2 = (0,1,0)
v3 = (0,0,1)
v4 = (0,1,-1)
v5 = (0,1,1)
v6 = (1,0,1)
v7 = (1,0,-1)
v8 = (1,1,0)
v9 = (1,-1,0)
v10 = (-1,1,1)
v11 = (1,-1,1)
v12 = (1,1,-1)
v13 = (1,1,1)
v14 = (2,1,1)
v15 = (2,-1,1)
v16 = (-2,1,1)
v17 = (2,1,-1)
v18 = (1,2,-1)
v19 = (1,1,2)
v20 = (-1,2,1)
v21 = (1,1,-2) 
v22 = (1,-1,2)
v23 = (1,-2,1)
v24 = (1,2,1) 
v25 = (-1,1,2)
v26 = (1,0,2)
v27 = (2,-1,0)
v28 = (2,0,-1)
v29 = (1,2,0)
v30 = (1,-2,0)
v31 = (2,0,1)
v32 = (1,0,-2)
v33 = (2,1,0)

# Create dictionary of all vectors
vectors = {f'v{i}': eval(f'v{i}') for i in range(1, 34)}

# Find all orthogonal pairs
orthogonal_pairs = []
for (name1, vec1), (name2, vec2) in itertools.combinations(vectors.items(), 2):
    dot_product = sum(a*b for a,b in zip(vec1, vec2))
    if dot_product == 0:
        orthogonal_pairs.append((name1, name2))

# Create and visualize the graph
G = nx.Graph()

# Add nodes with vector coordinates as attributes
for vector_name, coords in vectors.items():
    G.add_node(vector_name, coords=coords)

# Add edges based on orthogonal pairs
G.add_edges_from(orthogonal_pairs)

print(f"Graph created: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
print(f"Orthogonal pairs found: {len(orthogonal_pairs)}")

# Get adjacency matrix and print upper triangle column by column
adj_matrix = nx.adjacency_matrix(G).toarray()
n = len(adj_matrix)

print(f"Adjacency matrix size: {n}x{n}")

# Extract upper triangle column by column
upper_triangle_cols = []
for j in range(n):  # for each column
    col_values = []
    for i in range(j):  # only rows above diagonal
        col_values.append(str(adj_matrix[i][j]))
    if col_values:  # only add if there are values (skip first column which is empty)
        upper_triangle_cols.append(''.join(col_values))

# Print the result
result = ''.join(upper_triangle_cols)
print("Upper triangle representation:")
print(result)
print(f"Total length: {len(result)}")

# Create graph6 string representation
graph6_string = nx.to_graph6_bytes(G, header=False).decode('ascii')
print("\nGraph6 string representation:")
print(graph6_string)

# Plot the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=1, iterations=50)

# Create labels with vector coordinates
labels = {node: f"{node}\n{vectors[node]}" for node in G.nodes()}

nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', 
        font_size=8, font_weight='bold', edge_color='gray',
        labels=labels)
plt.title("Orthogonality Graph of 3D Vectors")
plt.tight_layout()
plt.show()

# Test 010-colorability using test_010.py
print("\n=== Testing 010-Colorability ===")
print(f"Testing the 01-string with test_010.py...")

# Create a temporary file with the 01-string
temp_input_file = "temp_01_string.txt"
with open(temp_input_file, 'w') as f:
    f.write(f"Solution 1: {result}\n")

# Call test_010.py with CNF output file
test_010_path = "test_010.py"
cnf_output_file = "minimal_conflicts_010.cnf"
if os.path.exists(test_010_path):
    try:
        process = subprocess.Popen([sys.executable, test_010_path, temp_input_file, cnf_output_file], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("test_010.py output:")
            print(stdout)
            print(f"CNF formula saved to: {cnf_output_file}")
        else:
            print(f"Error running test_010.py (return code {process.returncode}):")
            print(stderr)
    except Exception as e:
        print(f"Failed to execute test_010.py: {str(e)}")
else:
    print(f"Error: {test_010_path} not found")

# Clean up temporary file
try:
    os.remove(temp_input_file)
except:
    pass

# Solve with CaDiCaL and generate proof
print("\n=== Solving with CaDiCaL and Generating Proof ===")

# Generate unique timestamp for file names
timestamp = int(time.time())
cnf_file = f"minimal_conflicts_010_{timestamp}.cnf"
drat_file = f"minimal_conflicts_010_{timestamp}.drat"
core_file = f"minimal_conflicts_010_{timestamp}.core"

# Copy the CNF file with unique name
cnf_files = [f for f in os.listdir('.') if f.startswith('minimal_conflicts_010') and f.endswith('.cnf')]
if cnf_files:
    # Use the most recent CNF file
    cnf_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    source_cnf = cnf_files[0]
    shutil.copy(source_cnf, cnf_file)
    print(f"Copied CNF file {source_cnf} to: {cnf_file}")
else:
    print("Error: No minimal_conflicts_010*.cnf files found")
    exit(1)

# Call CaDiCaL to solve and generate DRAT proof
cadical_path = "./cadical/build/cadical"
if os.path.exists(cadical_path):
    try:
        print(f"Running CaDiCaL: {cadical_path} {cnf_file} {drat_file}")
        process = subprocess.Popen([cadical_path, cnf_file, drat_file], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("CaDiCaL completed successfully")
            print("CaDiCaL output:")
            print(stdout)
        else:
            print(f"Error running CaDiCaL (return code {process.returncode}):")
            print(stderr)
    except Exception as e:
        print(f"Failed to execute CaDiCaL: {str(e)}")
else:
    print(f"Error: {cadical_path} not found")

# Call drat-trim to verify proof and extract core
drat_trim_path = "./drat-trim/drat-trim"
if os.path.exists(drat_trim_path):
    try:
        print(f"Running drat-trim: {drat_trim_path} {cnf_file} {drat_file} -c {core_file}")
        process = subprocess.Popen([drat_trim_path, cnf_file, drat_file, "-c", core_file], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("drat-trim completed successfully")
            print("drat-trim output:")
            print(stdout)
            print(f"Core file generated: {core_file}")
        else:
            print(f"Error running drat-trim (return code {process.returncode}):")
            print(stderr)
    except Exception as e:
        print(f"Failed to execute drat-trim: {str(e)}")
else:
    print(f"Error: {drat_trim_path} not found")

print(f"\n=== Summary of Generated Files ===")
print(f"CNF file: {cnf_file}")
print(f"DRAT proof: {drat_file}")
print(f"Core file: {core_file}")
print(f"Timestamp: {timestamp}")

# Construct graph from UNSAT core
print("\n=== Constructing Graph from UNSAT Core ===")

def parse_unsat_core(core_file):
    """Parse UNSAT core file and extract edge and triangle constraints"""
    edges = []
    triangles = []
    
    try:
        with open(core_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c') or line.startswith('p'):
                    continue
                
                # Parse clause
                clause = line.split()[:-1]  # Remove trailing '0'
                if not clause:
                    continue
                
                # Check if it's an edge constraint (all negative literals)
                if all(lit.startswith('-') for lit in clause):
                    if len(clause) == 2:
                        # Edge constraint: -u -v means edge between u and v
                        u = int(clause[0][1:])  # Remove the '-' and convert to int
                        v = int(clause[1][1:])
                        edges.append((u, v))
                
                # Check if it's a triangle constraint (all positive literals)
                elif all(not lit.startswith('-') for lit in clause):
                    if len(clause) == 3:
                        # Triangle constraint: u v w means edges (u,v), (v,w), (u,w)
                        u = int(clause[0])
                        v = int(clause[1])
                        w = int(clause[2])
                        # Add all three edges of the triangle
                        edges.append((u, v))
                        edges.append((v, w))
                        edges.append((u, w))
                        triangles.append((u, v, w))
    
    except Exception as e:
        print(f"Error reading core file {core_file}: {e}")
        return [], []
    
    return edges, triangles

# Parse the UNSAT core
core_edges, core_triangles = parse_unsat_core(core_file)
print(f"Found {len(core_edges)} edges and {len(core_triangles)} triangles in UNSAT core")

# Remove duplicate edges (since triangles add multiple edges)
unique_edges = list(set(core_edges))
print(f"Unique edges after deduplication: {len(unique_edges)}")

# Create graph from UNSAT core
G_core = nx.Graph()
G_core.add_edges_from(unique_edges)

print(f"Core graph: {G_core.number_of_nodes()} vertices, {G_core.number_of_edges()} edges")

# Generate graph6 string for core graph
core_graph6_string = nx.to_graph6_bytes(G_core, header=False).decode('ascii')
print(f"\nCore graph6 string: {core_graph6_string}")

# Plot the core graph
plt.figure(figsize=(12, 8))
pos_core = nx.spring_layout(G_core, k=1, iterations=50)

# Create labels for vertices (using vertex numbers)
labels_core = {node: str(node) for node in G_core.nodes()}

nx.draw(G_core, pos_core, with_labels=True, node_size=800, node_color='lightcoral', 
        font_size=10, font_weight='bold', edge_color='red',
        labels=labels_core)
plt.title(f"Graph from UNSAT Core ({G_core.number_of_nodes()} vertices, {G_core.number_of_edges()} edges)")
plt.tight_layout()
plt.show()

# Show edge and triangle details
print(f"\n=== Core Graph Details ===")
print("Edges in core graph:")
for edge in sorted(unique_edges):
    print(f"  {edge[0]} -- {edge[1]}")

if core_triangles:
    print("\nTriangles in core graph:")
    for triangle in core_triangles:
        print(f"  {triangle[0]} -- {triangle[1]} -- {triangle[2]}")

# Compare original graph with core graph
print(f"\n=== Graph Comparison Statistics ===")
print(f"Original orthogonality graph:")
print(f"  Vertices: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Edge density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.4f}")

print(f"\nUNSAT core graph:")
print(f"  Vertices: {G_core.number_of_nodes()}")
print(f"  Edges: {G_core.number_of_edges()}")
if G_core.number_of_nodes() > 1:
    core_density = G_core.number_of_edges() / (G_core.number_of_nodes() * (G_core.number_of_nodes() - 1) / 2)
    print(f"  Edge density: {core_density:.4f}")
else:
    print(f"  Edge density: N/A (insufficient vertices)")

print(f"\nReduction statistics:")
print(f"  Edge reduction: {G.number_of_edges()} → {G_core.number_of_edges()} (removed {G.number_of_edges() - G_core.number_of_edges()} edges)")
print(f"  Edge reduction percentage: {((G.number_of_edges() - G_core.number_of_edges()) / G.number_of_edges() * 100):.2f}%")

if G_core.number_of_nodes() < G.number_of_nodes():
    print(f"  Vertex reduction: {G.number_of_nodes()} → {G_core.number_of_nodes()} (removed {G.number_of_nodes() - G_core.number_of_nodes()} vertices)")
    print(f"  Vertex reduction percentage: {((G.number_of_nodes() - G_core.number_of_nodes()) / G.number_of_nodes() * 100):.2f}%")

# Show which vertices from original graph are in core
core_vertices = set(G_core.nodes())
original_vertices = set(range(1, G.number_of_nodes() + 1))  # Assuming 1-based indexing
missing_vertices = original_vertices - core_vertices
print(f"\nVertices in core: {sorted(core_vertices)}")
if missing_vertices:
    print(f"Vertices removed: {sorted(missing_vertices)}")
else:
    print("All original vertices preserved in core")