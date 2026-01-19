import networkx as nx
from networkx import isomorphism
import matplotlib.pyplot as plt
import numpy as np
import itertools
import importlib.util
import sys

# Import the graph creation functions from SI-C.py
# Load the SI-C.py module dynamically
import os
spec = importlib.util.spec_from_file_location("SI_C", os.path.join(os.path.dirname(__file__), "SI-C.py"))
SI_C = importlib.util.module_from_spec(spec)
sys.modules["SI_C"] = SI_C
spec.loader.exec_module(SI_C)

# Now we can access the functions
create_minimal_sic_graph = SI_C.create_minimal_sic_graph
create_complete_sic_graph = SI_C.create_complete_sic_graph
create_intermediate_sic_graph = SI_C.create_intermediate_sic_graph

# Vector definitions
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
v20 = (-1,2,1) #remove
v21 = (1,1,-2) #remove
v22 = (1,-1,2) #remove
v23 = (1,-2,1)
v24 = (1,2,1) #remove
v25 = (-1,1,2)

def dot_product(v, w):
    """Calculate the dot product of two vectors"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def are_orthogonal(v, w):
    """Check if two vectors are orthogonal"""
    return dot_product(v, w) == 0

def create_orthogonality_graph(vectors):
    """Create a graph where vertices are vectors and edges connect orthogonal vectors"""
    G = nx.Graph()
    
    # Add nodes with vector values as attributes
    for i, v in enumerate(vectors, 1):
        G.add_node(i, vector=v)
    
    # Add edges between orthogonal vectors
    for i, v in enumerate(vectors, 1):
        for j, w in enumerate(vectors, 1):
            if i < j and are_orthogonal(v, w):
                G.add_edge(i, j)
    
    return G

def draw_graph(G, title):
    """Draw a graph with a title"""
    plt.figure(figsize=(12, 10))
    
    # Try to use Kamada-Kawai layout if scipy is available, otherwise fall back to spring layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except ImportError:
        pos = nx.spring_layout(G, seed=42, k=0.5)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_weight='bold')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def check_isomorphism(G1, G2, name1, name2):
    """Check if two graphs are isomorphic"""
    GM = isomorphism.GraphMatcher(G1, G2)
    is_isomorphic = GM.is_isomorphic()
    
    print(f"Is {name1} isomorphic to {name2}? {is_isomorphic}")
    
    if is_isomorphic:
        print(f"Node mapping from {name1} to {name2}:")
        mapping = GM.mapping
        for node1, node2 in sorted(mapping.items()):
            print(f"  {node1} -> {node2}")
    
    return is_isomorphic

def find_intermediate_vector_subset():
    """
    Find a subset of the 25 vectors that creates an orthogonality graph
    isomorphic to the intermediate SI-C graph.
    """
    # Get the intermediate SI-C graph
    intermediate_sic = create_intermediate_sic_graph()
    
    # Ensure node labels start from 1 to match our vector indexing
    intermediate_sic = nx.convert_node_labels_to_integers(intermediate_sic, first_label=1)
    
    # Get the number of nodes in the intermediate graph
    num_nodes_intermediate = intermediate_sic.number_of_nodes()
    print(f"Intermediate SI-C graph has {num_nodes_intermediate} nodes and {intermediate_sic.number_of_edges()} edges")
    
    # Create all vectors
    all_vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                  v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25]
    
    # First try: check if the first N vectors form the intermediate graph
    # (where N is the number of nodes in the intermediate graph)
    first_n_vectors = all_vectors[:num_nodes_intermediate]
    first_n_graph = create_orthogonality_graph(first_n_vectors)
    
    print(f"Testing first {num_nodes_intermediate} vectors...")
    is_isomorphic = check_isomorphism(first_n_graph, intermediate_sic, 
                                     f"First {num_nodes_intermediate} vectors graph", 
                                     "Intermediate SI-C")
    
    if is_isomorphic:
        return first_n_vectors
    
    # If that doesn't work, we'll try a more systematic approach
    print("\nFirst N vectors don't match. Trying to find a matching subset...")
    
    # Try different subsets of vectors
    # Start with the minimal SI-C vectors (first 13) and add others
    base_vectors = all_vectors[:13]  # Start with minimal SI-C vectors
    remaining_vectors = all_vectors[13:]
    
    # Calculate how many additional vectors we need
    additional_needed = num_nodes_intermediate - 13
    
    if additional_needed <= 0:
        print("Error: Intermediate SI-C has fewer nodes than minimal SI-C?")
        return None
    
    print(f"Need to find {additional_needed} vectors to add to the minimal 13")
    
    # Try different combinations of additional vectors
    for combo in itertools.combinations(remaining_vectors, additional_needed):
        test_vectors = base_vectors + list(combo)
        test_graph = create_orthogonality_graph(test_vectors)
        
        # Check if this combination creates a graph isomorphic to intermediate SI-C
        GM = isomorphism.GraphMatcher(test_graph, intermediate_sic)
        if GM.is_isomorphic():
            print("\nFound a matching subset!")
            # Create vector indices for reporting
            vector_indices = list(range(1, 14)) + [14 + remaining_vectors.index(v) for v in combo]
            print(f"Vector indices: {vector_indices}")
            return test_vectors
    
    print("Could not find a matching subset of vectors.")
    return None

def find_minimal_intermediate_subset():
    """
    Find a subset of the 25 vectors by removing as few vectors as possible,
    such that the resulting orthogonality graph:
    1. Contains the minimal SI-C graph as a subgraph
    2. Is itself a subgraph of the complete SI-C graph
    3. Is itself a subgraph of CK-31
    """
    # Get the reference graphs
    minimal_sic = create_minimal_sic_graph()
    complete_sic = create_complete_sic_graph()
    ck31_graph = SI_C.create_ck31_graph()  # Import CK-31 graph
    
    # Ensure node labels start from 1
    minimal_sic = nx.convert_node_labels_to_integers(minimal_sic, first_label=1)
    complete_sic = nx.convert_node_labels_to_integers(complete_sic, first_label=1)
    ck31_graph = nx.convert_node_labels_to_integers(ck31_graph, first_label=1)
    
    # Create all vectors
    all_vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                  v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25]
    
    # Start with all 25 vectors and try removing one at a time
    valid_solutions = []
    
    print("Searching for minimal intermediate subset by removing vectors one by one...")
    
    # Try removing each vector individually first
    for i in range(13, 25):  # Only try removing vectors beyond the minimal 13
        test_vectors = all_vectors.copy()
        removed_vector = test_vectors.pop(i)
        
        test_graph = create_orthogonality_graph(test_vectors)
        
        # Check if minimal_sic is a subgraph of test_graph
        GM1 = isomorphism.GraphMatcher(test_graph, minimal_sic)
        minimal_is_subgraph = GM1.subgraph_is_monomorphic()
        
        # Check if test_graph is a subgraph of complete_sic
        GM2 = isomorphism.GraphMatcher(complete_sic, test_graph)
        is_subgraph_of_complete = GM2.subgraph_is_monomorphic()
        
        # Check if test_graph is a subgraph of CK-31
        GM3 = isomorphism.GraphMatcher(ck31_graph, test_graph)
        is_subgraph_of_ck31 = GM3.subgraph_is_monomorphic()
        
        if minimal_is_subgraph and is_subgraph_of_complete and is_subgraph_of_ck31:
            print(f"Found valid intermediate by removing vector v{i+1}: {removed_vector}")
            valid_solutions.append({"removed": [i+1], "vectors": test_vectors})
    
    # If no single removal works, try removing pairs
    if len(valid_solutions) == 0:
        print("No single vector removal works. Trying pairs...")
        
        for i, j in itertools.combinations(range(13, 25), 2):
            test_vectors = all_vectors.copy()
            # Remove in reverse order to avoid index shifting
            if j > i:
                removed_vector2 = test_vectors.pop(j)
                removed_vector1 = test_vectors.pop(i)
            else:
                removed_vector1 = test_vectors.pop(i)
                removed_vector2 = test_vectors.pop(j)
            
            test_graph = create_orthogonality_graph(test_vectors)
            
            # Check if minimal_sic is a subgraph of test_graph
            GM1 = isomorphism.GraphMatcher(test_graph, minimal_sic)
            minimal_is_subgraph = GM1.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of complete_sic
            GM2 = isomorphism.GraphMatcher(complete_sic, test_graph)
            is_subgraph_of_complete = GM2.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of CK-31
            GM3 = isomorphism.GraphMatcher(ck31_graph, test_graph)
            is_subgraph_of_ck31 = GM3.subgraph_is_monomorphic()
            
            if minimal_is_subgraph and is_subgraph_of_complete and is_subgraph_of_ck31:
                print(f"Found valid intermediate by removing vectors v{i+1} and v{j+1}")
                valid_solutions.append({"removed": [i+1, j+1], "vectors": test_vectors})
    
    # If still no success, try removing triplets
    if len(valid_solutions) == 0:
        print("No pair removal works. Trying triplets...")
        
        for i, j, k in itertools.combinations(range(13, 25), 3):
            test_vectors = all_vectors.copy()
            # Remove in reverse order to avoid index shifting
            test_vectors.pop(max(i, j, k))
            test_vectors.pop(sorted([i, j, k])[1])
            test_vectors.pop(min(i, j, k))
            
            test_graph = create_orthogonality_graph(test_vectors)
            
            # Check if minimal_sic is a subgraph of test_graph
            GM1 = isomorphism.GraphMatcher(test_graph, minimal_sic)
            minimal_is_subgraph = GM1.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of complete_sic
            GM2 = isomorphism.GraphMatcher(complete_sic, test_graph)
            is_subgraph_of_complete = GM2.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of CK-31
            GM3 = isomorphism.GraphMatcher(ck31_graph, test_graph)
            is_subgraph_of_ck31 = GM3.subgraph_is_monomorphic()
            
            if minimal_is_subgraph and is_subgraph_of_complete and is_subgraph_of_ck31:
                print(f"Found valid intermediate by removing vectors v{i+1}, v{j+1}, and v{k+1}")
                valid_solutions.append({"removed": [i+1, j+1, k+1], "vectors": test_vectors})
    
    # If still no success, try removing quadruplets
    if len(valid_solutions) == 0:
        print("No triplet removal works. Trying quadruplets...")
        
        for i, j, k, l in itertools.combinations(range(13, 25), 4):
            test_vectors = all_vectors.copy()
            # Remove in reverse order to avoid index shifting
            indices = sorted([i, j, k, l], reverse=True)
            for idx in indices:
                test_vectors.pop(idx)
            
            test_graph = create_orthogonality_graph(test_vectors)
            
            # Check if minimal_sic is a subgraph of test_graph
            GM1 = isomorphism.GraphMatcher(test_graph, minimal_sic)
            minimal_is_subgraph = GM1.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of complete_sic
            GM2 = isomorphism.GraphMatcher(complete_sic, test_graph)
            is_subgraph_of_complete = GM2.subgraph_is_monomorphic()
            
            # Check if test_graph is a subgraph of CK-31
            GM3 = isomorphism.GraphMatcher(ck31_graph, test_graph)
            is_subgraph_of_ck31 = GM3.subgraph_is_monomorphic()
            
            if minimal_is_subgraph and is_subgraph_of_complete and is_subgraph_of_ck31:
                print(f"Found valid intermediate by removing vectors v{i+1}, v{j+1}, v{k+1}, and v{l+1}")
                valid_solutions.append({"removed": [i+1, j+1, k+1, l+1], "vectors": test_vectors})
    
    # Report results
    if valid_solutions:
        print(f"\nFound {len(valid_solutions)} valid intermediate subsets:")
        
        for idx, solution in enumerate(valid_solutions, 1):
            removed = solution["removed"]
            vectors = solution["vectors"]
            
            print(f"\nSolution {idx}: Removed {len(removed)} vectors: {removed}")
            print(f"Resulting graph has {len(vectors)} vectors")
            
            # Create and analyze the resulting graph
            intermediate_graph = create_orthogonality_graph(vectors)
            print(f"Intermediate graph has {intermediate_graph.number_of_nodes()} nodes and {intermediate_graph.number_of_edges()} edges")
            
            # Compare with minimal and complete
            minimal_edges = minimal_sic.number_of_edges()
            complete_edges = complete_sic.number_of_edges()
            ck31_edges = ck31_graph.number_of_edges()
            intermediate_edges = intermediate_graph.number_of_edges()
            
            print(f"Minimal SI-C has {minimal_edges} edges")
            print(f"Complete SI-C has {complete_edges} edges")
            print(f"CK-31 has {ck31_edges} edges")
            print(f"Intermediate has {intermediate_edges} edges")
            print(f"Added {intermediate_edges - minimal_edges} edges to minimal")
            print(f"Removed {complete_edges - intermediate_edges} edges from complete")
            print(f"Removed {ck31_edges - intermediate_edges} edges from CK-31")
        
        # Return the first solution (with fewest removals)
        return valid_solutions[0]["vectors"]
    else:
        print("Could not find a valid intermediate subset by removing up to 4 vectors.")
        return None

def main():
    """Main function to create and check orthogonality graphs"""
    # Create vectors list
    all_vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                  v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25]
    
    # Create orthogonality graphs
    minimal_vectors = all_vectors[:13]  # First 13 vectors (v1-v13)
    complete_vectors = all_vectors      # All 25 vectors (v1-v25)
    
    minimal_ortho_graph = create_orthogonality_graph(minimal_vectors)
    complete_ortho_graph = create_orthogonality_graph(complete_vectors)
    
    # Print graph statistics
    print(f"Minimal orthogonality graph: {minimal_ortho_graph.number_of_nodes()} nodes, {minimal_ortho_graph.number_of_edges()} edges")
    print(f"Complete orthogonality graph: {complete_ortho_graph.number_of_nodes()} nodes, {complete_ortho_graph.number_of_edges()} edges")
    
    # Create reference SI-C graphs by importing from SI-C.py
    minimal_sic = create_minimal_sic_graph()
    complete_sic = create_complete_sic_graph()
    
    print(f"Minimal SI-C graph: {minimal_sic.number_of_nodes()} nodes, {minimal_sic.number_of_edges()} edges")
    print(f"Complete SI-C graph: {complete_sic.number_of_nodes()} nodes, {complete_sic.number_of_edges()} edges")
    
    # Check isomorphism
    print("\nChecking isomorphism between orthogonality graphs and SI-C graphs:")
    check_isomorphism(minimal_ortho_graph, minimal_sic, "Minimal Orthogonality Graph", "Minimal SI-C")
    check_isomorphism(complete_ortho_graph, complete_sic, "Complete Orthogonality Graph", "Complete SI-C")
    
    # Find minimal intermediate subset
    print("\n=== Finding Minimal Intermediate Subset ===")
    minimal_intermediate_vectors = find_minimal_intermediate_subset()
    
    if minimal_intermediate_vectors:
        minimal_intermediate_graph = create_orthogonality_graph(minimal_intermediate_vectors)
        #draw_graph(minimal_intermediate_graph, "Minimal Intermediate Orthogonality Graph")
        
        # Print the vectors used
        print("\nVectors used for Minimal Intermediate Graph:")
        for i, v in enumerate(minimal_intermediate_vectors, 1):
            print(f"v{i}: {v}")
    
    # Find vectors for intermediate SI-C
    print("\n=== Finding vectors for Intermediate SI-C Graph ===")
    intermediate_vectors = find_intermediate_vector_subset()
    
    if intermediate_vectors:
        intermediate_ortho_graph = create_orthogonality_graph(intermediate_vectors)
        #draw_graph(intermediate_ortho_graph, "Intermediate Orthogonality Graph")
        
        # Print the vectors used
        print("\nVectors used for Intermediate SI-C:")
        for i, v in enumerate(intermediate_vectors, 1):
            print(f"v{i}: {v}")
    
    # Draw the original graphs
    draw_graph(minimal_ortho_graph, "Minimal Orthogonality Graph (v1-v13)")
    draw_graph(complete_ortho_graph, "Complete Orthogonality Graph (v1-v25)")

if __name__ == "__main__":
    main()
