"""v1 = (1,0,0)
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
v13 = (1,1,1)"""

v1 = (-1,1,1)
v2 = (1,-1,1)
v3 = (1,1,-1)
v4 = (1,1,1)
v5 = (1,0,0)
v6 = (0,1,0)
v7 = (0,0,1)
v8 = (1,-1,0)
v9 = (1,0,-1)
v10 = (0,1,-1)
v11 = (0,1,1)
v12 = (1,0,1)
v13 = (1,1,0)

# Function to calculate dot product of two vectors
def dot_product(vec1, vec2):
    return sum(a*b for a, b in zip(vec1, vec2))

# Function to check if two vectors are orthogonal
def is_orthogonal(vec1, vec2):
    return dot_product(vec1, vec2) == 0

# List of all vectors
vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
vector_names = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"]

# Create orthogonality graph
import networkx as nx
import matplotlib.pyplot as plt
import subprocess

# Create a graph
G = nx.Graph()
G.add_nodes_from(vector_names)

# Add edges for orthogonal pairs
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        if is_orthogonal(vectors[i], vectors[j]):
            G.add_edge(vector_names[i], vector_names[j])

# Visualize the graph
plt.figure(figsize=(10, 8))
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.title("Orthogonality Graph of 3D Vectors", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# Print graph statistics
print(f"\nGraph Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Get sorted list of nodes
nodes = list(G.nodes())
n = len(nodes)

# Create upper triangle as a continuous string (excluding diagonals)
print("\nUpper triangle of adjacency matrix as a string (excluding diagonals):")
result = ""
for j in range(n):  # Iterate through columns
    print(f"\nProcessing column {j+1} (node {nodes[j]}):")
    for i in range(j):  # Iterate through rows above the diagonal
        edge_exists = G.has_edge(nodes[i], nodes[j])
        result += "1" if edge_exists else "0"
        print(f"  Checking row {i+1} (node {nodes[i]}): {'1' if edge_exists else '0'}")

print("\nFinal 01-string:")
print(result)

# Call RCL with the 01-string
RCL_PATH = "./nauty2_8_8/RCL"
print(f"\nCalling RCL with the 01-string")
try:
    process = subprocess.Popen([RCL_PATH, result], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print("RCL output:")
        print(stdout)
        
        # Extract the output string from RCL
        output_string = stdout.split("Output string: ")[-1].strip()
        
        # Convert to variables (optional)
        variables = []
        for i, char in enumerate(output_string[:n*(n-1)//2], start=1):
            variables.append(str(i) if char == "1" else f"-{i}")
        variables_str = " ".join(variables)
        
        print("\nVariables:")
        print(variables_str)
        
    else:
        print(f"Error running RCL (return code {process.returncode}):")
        print(stderr)
except Exception as e:
    print(f"Failed to execute RCL: {str(e)}")

# Optionally, create a new graph from the canonical labeling
if 'output_string' in locals():
    print("\nCreating graph from canonical labeling...")
    canonical_graph = nx.Graph()
    canonical_graph.add_nodes_from(range(1, n+1))
    
    # Add edges based on the canonical labeling
    edge_index = 0
    for j in range(1, n+1):
        for i in range(1, j):
            if edge_index < len(output_string) and output_string[edge_index] == "1":
                canonical_graph.add_edge(i, j)
            edge_index += 1
    
    # Visualize the canonical graph
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(canonical_graph)
    nx.draw_networkx_nodes(canonical_graph, pos, node_size=700, node_color='lightgreen', alpha=0.8)
    nx.draw_networkx_edges(canonical_graph, pos, width=1.5, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(canonical_graph, pos, font_size=12, font_weight='bold')
    plt.title("Canonical Version of Orthogonality Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()