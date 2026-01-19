def are_colinear(a, b):
    """Check if two vectors are colinear using cross product"""
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ) == (0, 0, 0)

def are_orthogonal(a, b):
    """Check if two vectors are orthogonal using dot product"""
    return sum(x * y for x, y in zip(a, b)) == 0

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

vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
           v16, v17, v18, v19, v20, v21, v22, v23, v24, v25]

def is_complete_basis(vectors):
    """
    Check if a set of vectors forms a complete basis (3 mutually orthogonal vectors).
    
    Args:
        vectors: List of 3D vectors
        
    Returns:
        bool: True if vectors form a complete basis, False otherwise
    """
    if len(vectors) != 3:
        return False
    
    # Check that all pairs of vectors are orthogonal
    return (are_orthogonal(vectors[0], vectors[1]) and 
            are_orthogonal(vectors[0], vectors[2]) and 
            are_orthogonal(vectors[1], vectors[2]))

def find_orthogonal_vectors(initial_vectors, rounds=1):
    # Remove initial colinear duplicates
    unique_vectors = []
    for vec in initial_vectors:
        if not any(are_colinear(vec, existing) for existing in unique_vectors):
            unique_vectors.append(vec)

    existing = set(unique_vectors)
    cross_products = set()
    round_summary = []

    for round_num in range(1, rounds + 1):
        current_round = set()
        # Generate cross products from current unique vectors
        for i in range(len(unique_vectors)):
            for j in range(len(unique_vectors)):
                if i == j:
                    continue
                a, b = unique_vectors[i], unique_vectors[j]
                cp = (
                    a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0]
                )
                if cp != (0, 0, 0):
                    # Check against all existing vectors
                    if not any(are_colinear(cp, e) for e in existing):
                        # Check against current round's cross products
                        if not any(are_colinear(cp, ccp) for (ccp, _, _) in current_round):
                            current_round.add((cp, a, b))
        # Add to main results and update existing vectors
        for cp_info in current_round:
            cp, a, b = cp_info
            existing.add(cp)
            cross_products.add(cp_info)
        # Write round vectors to file
        with open(f"round_{round_num}_vectors.txt", "w") as f:
            for vec_info in sorted(current_round, key=lambda x: x[0]):
                cp, a, b = vec_info
                f.write(f"{cp} (orthogonal to {a} and {b})\n")
        # Record summary for this round
        round_summary.append(len(current_round))
        # Prepare unique_vectors for next round
        unique_vectors = []
        for vec in existing:
            if not any(are_colinear(vec, u) for u in unique_vectors):
                unique_vectors.append(vec)

    # Print summary of vectors added per round
    for i, count in enumerate(round_summary, start=1):
        print(f"Round {i}: Added {count} vectors")

    # Post-processing: Keep only vectors that are part of at least one complete basis
    vectors_in_bases = set()
    for i, vec1 in enumerate(unique_vectors):
        for j, vec2 in enumerate(unique_vectors):
            if i == j:
                continue
            if not are_orthogonal(vec1, vec2):
                continue
            for k, vec3 in enumerate(unique_vectors):
                if i == k or j == k:
                    continue
                if are_orthogonal(vec1, vec3) and are_orthogonal(vec2, vec3):
                    vectors_in_bases.add(vec1)
                    vectors_in_bases.add(vec2)
                    vectors_in_bases.add(vec3)
    
    # Filter out vectors not in any complete basis
    filtered_vectors = [v for v in unique_vectors if v in vectors_in_bases]
    
    print(f"Removed {len(unique_vectors) - len(filtered_vectors)} vectors that weren't part of any complete basis")
    print(f"Remaining vectors: {len(filtered_vectors)}")
    
    # Build orthogonality graph with filtered vectors
    # Ensure v1-v25 are the first 25 vertices in the same order
    all_vectors = []
    # First add v1 to v25 in order (if they exist in our set of vectors)
    for i in range(25):
        vec_name = f"v{i+1}"
        if vec_name in globals() and globals()[vec_name] in vectors_in_bases:
            all_vectors.append(globals()[vec_name])
    
    # Then add any remaining vectors
    for vec in filtered_vectors:
        if vec not in all_vectors:
            all_vectors.append(vec)
    
    graph = []
    for i, vec1 in enumerate(all_vectors):
        neighbors = []
        for j, vec2 in enumerate(all_vectors):
            if i != j and are_orthogonal(vec1, vec2):
                neighbors.append(j)
        graph.append(neighbors)

    # Write graph to LAD format
    with open("orthogonality_graph.lad", "w") as f:
        f.write(f"{len(all_vectors)}\n")  # Number of vertices
        for i, neighbors in enumerate(graph):
            # Label first 25 vertices as "0", the rest as "1"
            label = "0" if i < 25 else "1"
            f.write(f"{label} {len(neighbors)} {' '.join(map(str, neighbors))}\n")

    # Return the graph data along with the vectors for visualization
    return cross_products, all_vectors, graph

def visualize_orthogonality_graph(vectors, graph, output_file="orthogonality_graph.png"):
    """
    Visualize the orthogonality graph using NetworkX and Matplotlib.
    
    Args:
        vectors: List of vectors represented in the graph
        graph: Adjacency list representation of the graph
        output_file: Path to save the visualization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, vec in enumerate(vectors):
        # First 25 vectors (if present) get special treatment
        is_original = i < 25
        G.add_node(i, vector=vec, is_original=is_original)
    
    # Add edges
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            G.add_edge(i, j)
    
    # Set up the figure with a more reasonable size
    plt.figure(figsize=(10, 10), dpi=150)
    
    # Create a custom colormap for node colors
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for original, orange for derived
    node_colors = [colors[0] if G.nodes[n]['is_original'] else colors[1] for n in G.nodes()]
    
    # Use a force-directed layout for better spacing with many nodes
    if len(G) > 200:
        pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Draw nodes with different sizes based on type (smaller overall)
    node_sizes = [150 if G.nodes[n]['is_original'] else 50 for n in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4)
    
    # Add labels for original vectors only to avoid clutter
    labels = {i: f"v{i+1}" for i in range(min(13, len(vectors))) if i < len(vectors)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Add title and other information
    plt.title(f"Orthogonality Graph ({len(G)} vertices, {G.number_of_edges()} edges)", fontsize=12)
    plt.text(0.02, 0.02, "Blue: Original vectors\nOrange: Derived vectors", 
             transform=plt.gca().transAxes, fontsize=8, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to {output_file}")
    
    # Show statistics
    print(f"Graph has {len(G)} vertices and {G.number_of_edges()} edges")
    print(f"Average degree: {2 * G.number_of_edges() / len(G):.2f}")
    
    # Optional: show the plot
    plt.show()

def analyze_graph_distances(vectors, graph):
    """
    Analyze distances between derived (orange) vertices and their closest original (blue) vertices.
    
    Args:
        vectors: List of vectors represented in the graph
        graph: Adjacency list representation of the graph
        
    Returns:
        tuple: (min_distance, max_distance, avg_distance)
    """
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    original_nodes = []
    derived_nodes = []
    for i, vec in enumerate(vectors):
        is_original = i < 13
        G.add_node(i, vector=vec, is_original=is_original)
        if is_original:
            original_nodes.append(i)
        else:
            derived_nodes.append(i)
    
    # Add edges
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            G.add_edge(i, j)
    
    # For each derived node, find the shortest path to any original node
    derived_to_original_distances = {}
    
    for deriv in derived_nodes:
        # Calculate shortest paths from this derived node to all other nodes
        distances = nx.single_source_shortest_path_length(G, deriv)
        
        # Find the minimum distance to any original node
        min_dist_to_original = float('inf')
        closest_original = None
        
        for orig in original_nodes:
            if orig in distances:
                if distances[orig] < min_dist_to_original:
                    min_dist_to_original = distances[orig]
                    closest_original = orig
        
        if closest_original is not None:
            derived_to_original_distances[deriv] = (min_dist_to_original, closest_original)
    
    # Calculate statistics
    if not derived_to_original_distances:
        print("No paths found between derived and original nodes!")
        return None, None, None
    
    distances = [d[0] for d in derived_to_original_distances.values()]
    min_distance = min(distances)
    max_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    
    print(f"Minimum distance from a derived node to any original node: {min_distance}")
    print(f"Maximum distance from a derived node to any original node: {max_distance}")
    print(f"Average distance from derived nodes to their closest original node: {avg_distance:.2f}")
    
    # Find examples of min and max distances
    min_examples = []
    max_examples = []
    
    for deriv, (dist, orig) in derived_to_original_distances.items():
        if dist == min_distance and len(min_examples) < 3:
            min_examples.append((deriv, orig))
        if dist == max_distance and len(max_examples) < 3:
            max_examples.append((deriv, orig))
    
    print(f"Examples of derived nodes at minimum distance (derived→original): {min_examples}")
    print(f"Examples of derived nodes at maximum distance (derived→original): {max_examples}")
    
    # Count how many derived nodes are at each distance
    distance_counts = {}
    for dist in distances:
        distance_counts[dist] = distance_counts.get(dist, 0) + 1
    
    print("\nDistribution of distances:")
    for dist in sorted(distance_counts.keys()):
        print(f"  Distance {dist}: {distance_counts[dist]} derived nodes")
    
    return min_distance, max_distance, avg_distance

def check_vertices_in_triangles(graph, all_vectors):
    """
    Check if each vertex in the graph is part of at least one triangle.
    
    Args:
        graph: Adjacency list representation of the graph
        all_vectors: List of vectors represented in the graph
        
    Returns:
        tuple: (vertices_in_triangles, vertices_not_in_triangles)
    """
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, vec in enumerate(all_vectors):
        is_original = i < 13
        G.add_node(i, vector=vec, is_original=is_original)
    
    # Add edges
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            G.add_edge(i, j)
    
    # Check each vertex for triangles
    vertices_in_triangles = set()
    vertices_not_in_triangles = set()
    
    for node in G.nodes():
        in_triangle = False
        neighbors = list(G.neighbors(node))
        
        # Check if any two neighbors are connected
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if G.has_edge(neighbors[i], neighbors[j]):
                    in_triangle = True
                    break
            if in_triangle:
                break
        
        if in_triangle:
            vertices_in_triangles.add(node)
        else:
            vertices_not_in_triangles.add(node)
    
    # Separate original and derived vertices
    original_in_triangles = {v for v in vertices_in_triangles if v < 13}
    derived_in_triangles = {v for v in vertices_in_triangles if v >= 13}
    original_not_in_triangles = {v for v in vertices_not_in_triangles if v < 13}
    derived_not_in_triangles = {v for v in vertices_not_in_triangles if v >= 13}
    
    print(f"\nTriangle Analysis:")
    print(f"Original vertices in triangles: {len(original_in_triangles)}/{min(13, len(all_vectors))}")
    print(f"Derived vertices in triangles: {len(derived_in_triangles)}/{len(all_vectors) - min(13, len(all_vectors))}")
    print(f"Original vertices NOT in triangles: {len(original_not_in_triangles)}")
    print(f"Derived vertices NOT in triangles: {len(derived_not_in_triangles)}")
    
    if derived_not_in_triangles:
        print(f"\nDerived vertices not in any triangle: {sorted(derived_not_in_triangles)}")
        # Print the vectors for these vertices
        print("Corresponding vectors:")
        for idx in sorted(derived_not_in_triangles):
            print(f"  Vertex {idx}: {all_vectors[idx]}")
    
    return vertices_in_triangles, vertices_not_in_triangles

# Example usage with 2 rounds
rounds = 2
results, all_vectors, graph = find_orthogonal_vectors(vectors, rounds)

# Visualize the graph
visualize_orthogonality_graph(all_vectors, graph)

# After visualizing the graph, add this:
print("\nAnalyzing distances between derived and original vertices...")
min_dist, max_dist, avg_dist = analyze_graph_distances(all_vectors, graph)

# After analyzing distances, add this:
print("\nChecking which vertices are part of triangles...")
vertices_in_triangles, vertices_not_in_triangles = check_vertices_in_triangles(graph, all_vectors)

