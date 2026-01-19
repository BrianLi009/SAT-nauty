import itertools
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import shutil
import tempfile
from networkx.algorithms import isomorphism

def add_vars_to_cnf(input_cnf_file, variables, output_cnf_file=None):
    """
    Add variables as unit clauses to a CNF formula.
    
    Parameters:
        input_cnf_file (str): Path to the input CNF file
        variables (str): Space-separated string of variables (e.g., "1 -2 3")
        output_cnf_file (str): Path to the output CNF file. If None, overwrites input_cnf_file
    """
    # If no output file specified, overwrite input file
    if output_cnf_file is None:
        output_cnf_file = input_cnf_file
    
    # Convert variables to unit clauses
    unit_clauses = [int(var) for var in variables.split()]
    
    # Only read the header line to get the current clause count
    with open(input_cnf_file, 'r') as f:
        header_line = f.readline().strip()
        header_end_pos = f.tell()  # Remember where header ends
    
    # Parse header
    header = header_line.split()
    num_vars = int(header[2])
    num_clauses = int(header[3])
    
    # Create new header with updated clause count
    new_header = f"{header[0]} {header[1]} {num_vars} {num_clauses + len(unit_clauses)}\n"
    
    # Create new clauses
    new_clauses = [f"{var} 0\n" for var in unit_clauses]
    
    if output_cnf_file == input_cnf_file:
        # Modify in place: read header, update it, append new clauses
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_filename = temp_file.name
            
            # Write new header
            temp_file.write(new_header)
            
            # Copy rest of original file (skip header)
            with open(input_cnf_file, 'r') as original:
                original.readline()  # Skip header
                shutil.copyfileobj(original, temp_file)
            
            # Append new clauses
            temp_file.writelines(new_clauses)
        
        # Replace original file with temp file
        shutil.move(temp_filename, input_cnf_file)
    else:
        # Create new output file
        with open(output_cnf_file, 'w') as output_file:
            # Write new header
            output_file.write(new_header)
            
            # Copy rest of original file (skip header)
            with open(input_cnf_file, 'r') as input_file:
                input_file.readline()  # Skip header
                shutil.copyfileobj(input_file, output_file)
            
            # Append new clauses
            output_file.writelines(new_clauses)

def rcl_output_to_variables(output_string, num_nodes):
    """
    Convert RCL output string to positive/negative variables
    Only considers the first n choose 2 variables (upper triangle)
    
    Parameters:
        output_string (str): Binary string from RCL
        num_nodes (int): Number of nodes in the graph
    """
    # Calculate number of variables (n choose 2)
    num_vars = num_nodes * (num_nodes - 1) // 2
    
    # Only take the first num_vars characters
    relevant_string = output_string[:num_vars]
    
    # Debug: print parsed 01-string
    print(f"\nParsed 01-string (first {num_vars} characters):")
    print(relevant_string)
    
    # Convert to variables using to_var.py logic
    result = []
    for i, char in enumerate(relevant_string, start=1):
        result.append(str(i) if char == "1" else f"-{i}")
    return " ".join(result)

def is_edge_in_triangle(G, edge):
    """
    Check if an edge is part of a triangle in the graph.
    
    Parameters:
        G (nx.Graph): The graph to check
        edge (tuple): The edge to check (u, v)
    
    Returns:
        bool: True if the edge is part of a triangle, False otherwise
    """
    u, v = edge
    # Get common neighbors of the two vertices
    common_neighbors = set(G.neighbors(u)) & set(G.neighbors(v))
    return len(common_neighbors) > 0

def create_minimal_sic_graph():
    """Create the minimal SI-C graph with 13 nodes"""
    G = nx.Graph()
    
    # Add nodes
    for i in range(1, 14):
        G.add_node(i)
    
    # Add edges for minimal SI-C graph
    minimal_edges = [
        (1,2), (1,3), (1,4), (1,5), (2,3), (2,6), (2,7), 
        (3,8), (3,9), (4,5), (4,11), (4,12), (5,10), (5,13), 
        (6,7), (6,10), (6,12), (7,11), (7,13), (8,9), (8,10), (8,11), 
        (9,12), (9,13)
    ]
    G.add_edges_from(minimal_edges)
    
    return G

def create_complete_sic_graph():
    """Create the complete SI-C graph with 25 nodes"""
    G = nx.Graph()
    
    # Add all nodes from the diagram (1-25)
    for i in range(1, 26):
        G.add_node(i)
    
    # Add edges (treating dotted and solid lines the same)
    edges = [
        (1,2), (1,3), (1,4), (1,5), (2,3), (2,6), (2,7), (3,8), (3,9), 
        (4,5), (4,11), (4,12), (4,19), (4,25), (5,10), (5,14), (5,13), (5,20), 
        (6,7), (6,10), (6,12), (6,15), (6,23), (7,11), (7,13), (7,16), (7,21), 
        (8,9), (8,11), (8,10), (8,17), (8,24), (9,13), (9,12), (9,18), (9,22), 
        (10,15), (10,14), (10,24), (11,16), (11,17), (11,25), (12,18), (12,19), (12,23), 
        (13,20), (13,21), (13,22)
    ]
    G.add_edges_from(edges)
    
    return G

def create_ck31_graph():
    """Create the CK-31 graph from graph6 string"""
    CONWAY_GRAPH_STRING = "^xLAKA@G@?g?O?AgAc?e?BOA??C??A?C@?C?_??c??G?DGG?QAG?cCA?aCA?OCS?C@C_?OAc??OIG??"
    return nx.from_graph6_bytes(CONWAY_GRAPH_STRING.encode('ascii'))

def create_ck51_graph():
    """Create the CK-51 graph by adding triangles to CK-31"""
    # Start with CK-31
    conway_g = create_ck31_graph()
    
    # Add new vertices to make triangles
    new_vertex_id = max(conway_g.nodes()) + 1
    edges_to_check = list(conway_g.edges())  # Create a list to avoid modifying during iteration
    
    for edge in edges_to_check:
        if not is_edge_in_triangle(conway_g, edge):
            # Add new vertex and connect it to both endpoints of the edge
            conway_g.add_node(new_vertex_id)
            conway_g.add_edge(edge[0], new_vertex_id)
            conway_g.add_edge(edge[1], new_vertex_id)
            new_vertex_id += 1
    
    return conway_g

def create_intermediate_sic_graph():
    """Create the intermediate SI-C graph from graph6 string"""
    INTERMEDIATE_GRAPH_STRING = "T{dAH?`BOeAcAgAC?a?CO?Q?OC?CC?@C?CC?"
    return nx.from_graph6_bytes(INTERMEDIATE_GRAPH_STRING.encode('ascii'))

def count_triangles(G):
    """
    Count the number of triangles (cycles of length 3) in the graph.
    
    Parameters:
        G (nx.Graph): The graph to analyze
    
    Returns:
        int: Number of triangles in the graph
    """
    triangles = 0
    
    # For each node, check all pairs of its neighbors
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        
        # Check each pair of neighbors to see if they form a triangle
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                # If there's an edge between these neighbors, we have a triangle
                if G.has_edge(neighbors[i], neighbors[j]):
                    triangles += 1
    
    # Each triangle is counted 3 times (once for each vertex)
    return triangles // 3

def process_graph(G, graph_name, input_cnf="constraints_31_0_1", output_cnf=None):
    """
    Process a graph: draw it, analyze it, run RCL, and add to CNF
    
    Parameters:
        G (nx.Graph): The graph to process
        graph_name (str): Name of the graph for display
        input_cnf (str): Input CNF file path
        output_cnf (str): Output CNF file path (if None, will be input_cnf + "_" + graph_name)
    """
    if output_cnf is None:
        output_cnf = f"{input_cnf}_{graph_name.lower().replace('-', '_')}"
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # For SI-C graphs, use custom positions if it's the complete graph
    if graph_name == "Complete SI-C" or graph_name == "Minimal SI-C":
        # Set positions for the nodes to match the pentagram layout
        if 13 in G.nodes():  # Check if node 13 exists (center node)
            pos = {
                1: (0, 2),      # Top of triangle
                2: (-1.5, -1),  # Bottom left of triangle
                3: (1.5, -1),   # Bottom right of triangle
                4: (1, 1),      # Inner pentagon point
                5: (-1, 1),     # Inner pentagon point
                6: (-1.2, 0),   # Inner pentagon point
                7: (-0.5, -0.8),# Inner pentagon point
                8: (0.5, -0.8), # Inner pentagon point
                9: (1.2, 0),    # Inner pentagon point
                10: (-1.5, 0.5),# Inner connection
                11: (0, -1.5),  # Inner connection
                12: (1.5, 0.5), # Inner connection
                13: (0, 0),     # Center
            }
            
            # Add positions for outer nodes if they exist
            if graph_name == "Complete SI-C":
                outer_pos = {
                    14: (-1.8, 1.5),# Outer nodes
                    15: (-2.5, 0),
                    16: (-1.2, -2),
                    17: (0, -2.2),
                    18: (2.5, 0),
                    19: (1.8, 1.5),
                    20: (-0.5, 1.5),
                    21: (-0.5, -1.5),
                    22: (1.2, -0.3),
                    23: (0, 2.5),
                    24: (-2.8, -1),
                    25: (2.8, -1)
                }
                pos.update(outer_pos)
    
    nx.draw(G, pos, with_labels=True, node_color='white', node_size=500, edgecolors='black')
    plt.title(f"{graph_name} Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print graph statistics
    print(f"\n{graph_name} Graph Statistics:")
    print(f"- Number of vertices: {G.number_of_nodes()}")
    print(f"- Number of edges: {G.number_of_edges()}")
    
    # Count and print the number of triangles
    num_triangles = count_triangles(G)
    print(f"- Number of triangles: {num_triangles}")
    
    # Print degree of each vertex
    print(f"\nDegree of each vertex in {graph_name}:")
    for node in sorted(G.nodes()):
        degree = G.degree(node)
        print(f"Node {node}: {degree}")
    
    # Get sorted list of nodes
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    # Create upper triangle as a continuous string (excluding diagonals)
    print(f"\nGenerating upper triangle of adjacency matrix for {graph_name}:")
    result = ""
    for j in range(n):  # Iterate through columns
        for i in range(j):  # Iterate through rows above the diagonal
            edge_exists = G.has_edge(nodes[i], nodes[j])
            result += "1" if edge_exists else "0"
    
    print(f"\nFinal 01-string for {graph_name}:")
    print(result)
    
    # Call RCL with the 01-string
    RCL_PATH = "./nauty2_8_8/RCL"
    print(f"\nCalling RCL with the 01-string for {graph_name}")
    try:
        process = subprocess.Popen([RCL_PATH, result], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("RCL output:")
            print(stdout)
            
            # Extract the output string from RCL
            output_string = stdout.split("Output string: ")[-1].strip()
            
            # Convert to variables
            num_nodes = len(G.nodes())
            variables = rcl_output_to_variables(output_string, num_nodes)
            print("\nVariables:")
            print(variables)
            
            # Add variables to CNF file
            add_vars_to_cnf(input_cnf, variables, output_cnf)
            print(f"\nVariables added to CNF file: {output_cnf}")
        else:
            print(f"Error running RCL (return code {process.returncode}):")
            print(stderr)
    except Exception as e:
        print(f"Failed to execute RCL: {str(e)}")

# Original SI-C.py code for vector-based graph
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

# Create dictionary of all vectors
vectors = {f'v{i}': eval(f'v{i}') for i in range(1, 20)}

# Calculate orthogonal pairs
orthogonal_pairs = []
for (name1, vec1), (name2, vec2) in itertools.combinations(vectors.items(), 2):
    # Calculate dot product
    dot_product = sum(a*b for a,b in zip(vec1, vec2))
    if dot_product == 0:  # Vectors are orthogonal
        orthogonal_pairs.append((name1, name2))

def create_vector_graph():
    """Create graph from orthogonal vector pairs"""
    G = nx.Graph()
    
    # Create a list of all vectors (v1 through v25)
    vector_list = [
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
        v11, v12, v13, v14, v15, v16, v17, v18, v19, 
        v20, v21, v22, v23, v24, v25
    ]
    
    # Add nodes (vectors) with vector coordinates as labels
    for i, coords in enumerate(vector_list, 1):
        G.add_node(i, coords=coords)
    
    # Add edges for orthogonal pairs
    for i, vec1 in enumerate(vector_list, 1):
        for j, vec2 in enumerate(vector_list[i:], i + 1):
            # Calculate dot product
            dot_product = sum(a*b for a,b in zip(vec1, vec2))
            if dot_product == 0:  # Vectors are orthogonal
                G.add_edge(i, j)
    
    return G

def process_vector_graph(G, input_cnf="constraints_31_0_1", output_cnf="constraints_31_0_1_vector"):
    """Process the vector-based graph"""
    
    # Print vector mappings first
    print("\n=== Processing Vector-based Graph ===")
    print_vector_mappings()
    
    # Print graph statistics
    print(f"\nVector Graph Statistics:")
    print(f"- Number of vertices: {G.number_of_nodes()}")
    print(f"- Number of edges: {G.number_of_edges()}")
    
    # Count and print the number of triangles
    num_triangles = count_triangles(G)
    print(f"- Number of triangles: {num_triangles}")
    
    # Print degree of each vertex with its vector
    print(f"\nDegree of each vertex:")
    for node in sorted(G.nodes()):
        degree = G.degree(node)
        coords = G.nodes[node]['coords']
        print(f"Vertex {node:2d} {coords}: degree {degree}")
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create labels dictionary with vertex ID and coordinates
    labels = {node: f"{node}\n{G.nodes[node]['coords']}" for node in G.nodes()}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue',
            font_size=8, font_weight='bold', edge_color='gray', labels=labels)
    plt.title("Vector Orthogonality Graph\n(Vertices connected if their vectors are orthogonal)")
    plt.tight_layout()
    plt.show()
    
    # Get sorted list of nodes and create mapping
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    # Create mapping from sorted position to original vertex ID and vector
    print(f"\n=== BEFORE RCL: Vertex Ordering for Adjacency Matrix ===")
    print("Position in adjacency matrix -> Original Vertex ID -> Vector")
    print("-" * 65)
    position_to_vertex = {}
    position_to_vector = {}
    for pos, node in enumerate(nodes):
        coords = G.nodes[node]['coords']
        position_to_vertex[pos] = node
        position_to_vector[pos] = coords
        print(f"Position {pos+1:2d} -> Vertex {node:2d} -> {coords}")
    print("-" * 65)
    
    # Print upper triangle as a continuous string (excluding diagonals)
    print("\nGenerating upper triangle of adjacency matrix:")
    result = ""
    for j in range(n):  # Iterate through columns
        for i in range(j):  # Iterate through rows above the diagonal
            edge_exists = G.has_edge(nodes[i], nodes[j])
            result += "1" if edge_exists else "0"
    
    print(f"\nFinal 01-string for Vector Graph:")
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
            
            # Parse the canonical permutation from RCL output
            canonical_mapping = {}  # original_vertex -> canonical_position
            lines = stdout.split('\n')
            parsing_permutation = False
            
            for line in lines:
                if "Permutation (input -> canonical):" in line:
                    parsing_permutation = True
                    continue
                elif parsing_permutation and "->" in line:
                    parts = line.strip().split(" -> ")
                    if len(parts) == 2:
                        try:
                            original_vertex = int(parts[0])
                            canonical_position = int(parts[1])
                            canonical_mapping[original_vertex] = canonical_position
                        except ValueError:
                            continue
                elif parsing_permutation and line.strip() == "":
                    parsing_permutation = False
            
            # Extract the output string from RCL
            output_string = stdout.split("Output string: ")[-1].strip()
            
            # Parse RCL output to understand the canonical ordering
            print(f"\n=== AFTER RCL: Canonical Vertex Ordering ===")
            print("RCL has reordered vertices to canonical form")
            print("Original Vertex ID -> Canonical Position -> Vector")
            print("-" * 70)
            
            # Create reverse mapping: canonical_position -> original_vertex
            canonical_to_original = {}
            for orig, canon in canonical_mapping.items():
                canonical_to_original[canon] = orig
            
            # Show the canonical ordering
            for canon_pos in range(n):
                if canon_pos in canonical_to_original:
                    original_vertex_id = canonical_to_original[canon_pos]
                    # Map back to our vertex numbering (RCL uses 0-based, we use 1-based)
                    our_vertex_id = original_vertex_id + 1
                    if our_vertex_id <= len(nodes):
                        our_original_vertex = nodes[original_vertex_id]
                        vector_coords = G.nodes[our_original_vertex]['coords']
                        print(f"Vertex {our_original_vertex:2d} -> Position {canon_pos:2d} -> {vector_coords}")
                    else:
                        print(f"Vertex ? -> Position {canon_pos:2d} -> (mapping error)")
                else:
                    print(f"Position {canon_pos:2d} -> (no mapping found)")
            print("-" * 70)
            
            # Show vertex-to-vector mapping in canonical order
            print(f"\n=== Vertex-to-Vector Mapping (Canonical Order) ===")
            print("Canonical Position -> Original Vertex -> Vector")
            print("-" * 55)
            for canon_pos in range(n):
                if canon_pos in canonical_to_original:
                    original_vertex_id = canonical_to_original[canon_pos]
                    our_original_vertex = nodes[original_vertex_id]
                    vector_coords = G.nodes[our_original_vertex]['coords']
                    print(f"Position {canon_pos+1:2d} -> Vertex {our_original_vertex:2d} -> {vector_coords}")
                else:
                    print(f"Position {canon_pos+1:2d} -> (no mapping found)")
            print("-" * 55)
            
            # Convert to variables
            variables = rcl_output_to_variables(output_string, n)
            print("\nVariables from RCL:")
            print(variables)
            
            # Show interpretation of variables in terms of original vectors
            print(f"\n=== Variable Interpretation (Using Canonical Ordering) ===")
            print("Variable -> Edge -> Vector Pair")
            print("-" * 60)
            var_list = variables.split()
            var_index = 0
            
            for j in range(n):  # columns in canonical ordering
                for i in range(j):  # rows above diagonal in canonical ordering
                    if var_index < len(var_list):
                        var = var_list[var_index]
                        
                        # Map canonical positions back to original vertices
                        canon_vertex_i = i
                        canon_vertex_j = j
                        
                        if canon_vertex_i in canonical_to_original and canon_vertex_j in canonical_to_original:
                            orig_vertex_i = canonical_to_original[canon_vertex_i]
                            orig_vertex_j = canonical_to_original[canon_vertex_j]
                            
                            # Convert to our 1-based vertex IDs
                            our_vertex_i = orig_vertex_i + 1
                            our_vertex_j = orig_vertex_j + 1
                            
                            if our_vertex_i <= len(nodes) and our_vertex_j <= len(nodes):
                                # Get the actual vertex IDs from our sorted list
                                actual_vertex_i = nodes[orig_vertex_i]
                                actual_vertex_j = nodes[orig_vertex_j]
                                
                                vector_i = G.nodes[actual_vertex_i]['coords']
                                vector_j = G.nodes[actual_vertex_j]['coords']
                                
                                edge_status = "present" if var.startswith('-') else "absent"
                                var_display = var
                                
                                print(f"Var {var_display:3s} -> Edge({actual_vertex_i:2d},{actual_vertex_j:2d}) -> {vector_i} ↔ {vector_j} [{edge_status}]")
                            else:
                                print(f"Var {var:3s} -> Edge(?,?) -> (mapping error)")
                        else:
                            print(f"Var {var:3s} -> Edge(?,?) -> (canonical mapping not found)")
                        
                        var_index += 1
            print("-" * 60)
            
            # Summary of vertex-vector correspondences
            print(f"\n=== SUMMARY: Final Vertex-Vector Correspondences ===")
            print("After RCL canonicalization, each vertex maps to:")
            print("-" * 60)
            
            # Show all vertices and their vectors in original order for easy reference
            print("Original Input Order:")
            for node in sorted(G.nodes()):
                vector_coords = G.nodes[node]['coords']
                print(f"  Vertex {node:2d} -> Vector {vector_coords}")
            
            print("\nCanonical Order (after RCL processing):")
            for canon_pos in range(n):
                if canon_pos in canonical_to_original:
                    original_vertex_id = canonical_to_original[canon_pos]
                    our_original_vertex = nodes[original_vertex_id]
                    vector_coords = G.nodes[our_original_vertex]['coords']
                    print(f"  Position {canon_pos+1:2d} -> Vertex {our_original_vertex:2d} -> Vector {vector_coords}")
            print("-" * 60)
            
            # Add variables to CNF file
            add_vars_to_cnf(input_cnf, variables, output_cnf)
            print(f"\nVariables added to CNF file: {output_cnf}")
        else:
            print(f"Error running RCL (return code {process.returncode}):")
            print(stderr)
    except Exception as e:
        print(f"Failed to execute RCL: {str(e)}")

def check_all_subgraph_relations():
    """Check subgraph relations between all pairs of graphs"""
    print("\n=== Checking Subgraph Relations Between All Graph Pairs ===\n")
    
    # Create all graphs
    graphs = {
        "Minimal SI-C": create_minimal_sic_graph(),
        "Intermediate SI-C": create_intermediate_sic_graph(),
        "Complete SI-C": create_complete_sic_graph(),
        "CK-31": create_ck31_graph(),
        "CK-51": create_ck51_graph()
    }
    
    # Print graph statistics
    print("Graph Statistics:")
    for name, G in graphs.items():
        print(f"{name}: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    
    print("\nSubgraph Relations:")
    print("-" * 60)
    print(f"{'Graph A':<15} | {'Graph B':<15} | {'A is subgraph of B':<20} | {'B is subgraph of A':<20}")
    print("-" * 60)
    
    # Check all pairs
    for name1, G1 in graphs.items():
        for name2, G2 in graphs.items():
            if name1 != name2:
                # Check if G1 is a subgraph of G2
                GM1 = isomorphism.GraphMatcher(G2, G1)
                is_subgraph_1_of_2 = GM1.subgraph_is_monomorphic()
                
                # Check if G2 is a subgraph of G1
                GM2 = isomorphism.GraphMatcher(G1, G2)
                is_subgraph_2_of_1 = GM2.subgraph_is_monomorphic()
                
                print(f"{name1:<15} | {name2:<15} | {str(is_subgraph_1_of_2):<20} | {str(is_subgraph_2_of_1):<20}")
    
    print("-" * 60)

def print_vector_mappings():
    """Print the mapping between vertex IDs and their corresponding vectors"""
    print("\n=== Vector Mappings ===")
    print("Vertex ID -> Vector Coordinates")
    print("-" * 35)
    
    # Create a list of all vectors (v1 through v25)
    vector_list = [
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
        v11, v12, v13, v14, v15, v16, v17, v18, v19, 
        v20, v21, v22, v23, v24, v25
    ]
    
    for i, coords in enumerate(vector_list, 1):
        print(f"Vertex {i:2d} -> {coords}")
    
    print("-" * 35)
    print(f"Total vertices: {len(vector_list)}")

def main():
    """Main function to select and process a graph"""
    import sys
    
    # If no arguments, just do subgraph comparison
    if len(sys.argv) == 1:
        check_all_subgraph_relations()
        return
    
    # Check if command-line arguments are provided for graph processing
    if len(sys.argv) > 1:
        # Command-line mode
        if len(sys.argv) < 3:
            print("Usage: python3 SI-C.py n input_cnf [output_cnf]")
            print("  n: Graph choice (1-6)")
            print("  input_cnf: Input CNF file path")
            print("  output_cnf: (Optional) Output CNF file path")
            print("\nOr run without arguments to check subgraph relations between graphs")
            return
        
        choice = sys.argv[1]
        input_cnf = sys.argv[2]
        output_cnf = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        # Interactive mode
        print("Select a graph to process:")
        print("1. Minimal SI-C Graph (13 nodes)")
        print("2. Complete SI-C Graph (25 nodes)")
        print("3. CK-31 Graph (Conway Graph)")
        print("4. CK-51 Graph (Conway Graph with triangles)")
        print("5. Intermediate SI-C Graph")
        print("6. Vector-based Graph (orthogonal vectors)")
        
        choice = input("Enter your choice (1-6): ")
        input_cnf = input("Enter input CNF file path (default: constraints_31_0_1): ") or "constraints_31_0_1"
        output_cnf = None
    
    if choice == "1":
        G = create_minimal_sic_graph()
        process_graph(G, "Minimal SI-C", input_cnf, output_cnf)
    elif choice == "2":
        G = create_complete_sic_graph()
        process_graph(G, "Complete SI-C", input_cnf, output_cnf)
    elif choice == "3":
        G = create_ck31_graph()
        process_graph(G, "CK-31", input_cnf, output_cnf)
    elif choice == "4":
        G = create_ck51_graph()
        process_graph(G, "CK-51", input_cnf, output_cnf)
    elif choice == "5":
        G = create_intermediate_sic_graph()
        process_graph(G, "Intermediate SI-C", input_cnf, output_cnf)
    elif choice == "6":
        print_vector_mappings()
        G_vector = create_vector_graph()
        
        # IMPORTANT SANITY CHECK: Compare with Complete SI-C graph
        print("\n" + "="*60)
        print("SANITY CHECK: Comparing Vector Graph with Complete SI-C Graph")
        print("="*60)
        
        G_complete_sic = create_complete_sic_graph()
        
        print(f"Vector Graph: {G_vector.number_of_nodes()} vertices, {G_vector.number_of_edges()} edges")
        print(f"Complete SI-C Graph: {G_complete_sic.number_of_nodes()} vertices, {G_complete_sic.number_of_edges()} edges")
        
        # Check if they have the same number of vertices and edges
        if G_vector.number_of_nodes() != G_complete_sic.number_of_nodes():
            print("❌ MISMATCH: Different number of vertices!")
        elif G_vector.number_of_edges() != G_complete_sic.number_of_edges():
            print("❌ MISMATCH: Different number of edges!")
        else:
            # Check for isomorphism
            print("\nChecking for graph isomorphism...")
            GM = isomorphism.GraphMatcher(G_vector, G_complete_sic)
            is_isomorphic = GM.is_isomorphic()
            
            if is_isomorphic:
                print("✅ SUCCESS: Vector Graph is ISOMORPHIC to Complete SI-C Graph!")
                print("The vector-based construction correctly reproduces the SI-C structure.")
                
                # Show the mapping
                mapping = GM.mapping
                print(f"\nIsomorphism mapping (Vector vertex -> SI-C vertex):")
                for vec_node in sorted(mapping.keys())[:10]:  # Show first 10 mappings
                    sic_node = mapping[vec_node]
                    vec_coords = G_vector.nodes[vec_node]['coords']
                    print(f"  Vector {vec_node} {vec_coords} -> SI-C vertex {sic_node}")
                if len(mapping) > 10:
                    print(f"  ... and {len(mapping) - 10} more mappings")
            else:
                print("❌ FAILURE: Vector Graph is NOT isomorphic to Complete SI-C Graph!")
                print("This indicates a problem with either the vector construction or the SI-C definition.")
        
        print("="*60)
        
        process_vector_graph(G_vector, input_cnf, output_cnf or f"{input_cnf}_vector")
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()