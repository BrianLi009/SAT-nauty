import itertools
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from networkx.algorithms import isomorphism

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

def find_common_subgraph():
    """
    Find a graph that is:
    1. A subgraph of CK-31
    2. A subgraph of Complete SI-C
    3. Contains Minimal SI-C as a subgraph
    
    The goal is to find the largest possible graph satisfying these conditions.
    """
    # Create all three graphs
    ck31 = create_ck31_graph()
    complete_sic = create_complete_sic_graph()
    minimal_sic = create_minimal_sic_graph()
    
    print(f"CK-31: {ck31.number_of_nodes()} nodes, {ck31.number_of_edges()} edges")
    print(f"Complete SI-C: {complete_sic.number_of_nodes()} nodes, {complete_sic.number_of_edges()} edges")
    print(f"Minimal SI-C: {minimal_sic.number_of_nodes()} nodes, {minimal_sic.number_of_edges()} edges")
    
    # Relabel CK-31 to have sequential node IDs starting from 1
    ck31_relabeled = nx.convert_node_labels_to_integers(ck31, first_label=1)
    
    # Start with the Complete SI-C graph and remove nodes/edges until it becomes a subgraph of CK-31
    # while ensuring it still contains Minimal SI-C
    
    # First, check if Complete SI-C is already a subgraph of CK-31
    GM1 = isomorphism.GraphMatcher(ck31_relabeled, complete_sic)
    is_complete_subgraph_of_ck31 = GM1.subgraph_is_monomorphic()
    
    if is_complete_subgraph_of_ck31:
        print("Complete SI-C is already a subgraph of CK-31!")
        return complete_sic
    
    print("Complete SI-C is not a subgraph of CK-31. Finding a maximal common subgraph...")
    
    # Start with the minimal SI-C graph (which we know must be in our solution)
    common_graph = minimal_sic.copy()
    
    # Get all edges in Complete SI-C that are not in Minimal SI-C
    additional_edges = []
    for edge in complete_sic.edges():
        if edge not in minimal_sic.edges() and (edge[1], edge[0]) not in minimal_sic.edges():
            # Only consider edges between nodes that already exist in minimal_sic
            if edge[0] in minimal_sic.nodes() and edge[1] in minimal_sic.nodes():
                additional_edges.append(edge)
    
    print(f"Found {len(additional_edges)} additional edges in Complete SI-C that connect nodes in Minimal SI-C")
    
    # Try adding these edges one by one, checking if the graph remains a subgraph of CK-31
    edges_added = 0
    for edge in additional_edges:
        # Add the edge to our common graph
        common_graph.add_edge(*edge)
        
        # Check if it's still a subgraph of CK-31
        GM = isomorphism.GraphMatcher(ck31_relabeled, common_graph)
        if GM.subgraph_is_monomorphic():
            print(f"Added edge {edge}, graph is still a subgraph of CK-31")
            edges_added += 1
        else:
            # If not, remove the edge
            common_graph.remove_edge(*edge)
            print(f"Edge {edge} would make the graph not a subgraph of CK-31, skipping")
    
    print(f"Added {edges_added} edges to Minimal SI-C")
    
    # Now try adding nodes from Complete SI-C that are not in Minimal SI-C
    additional_nodes = [n for n in complete_sic.nodes() if n not in minimal_sic.nodes()]
    print(f"Found {len(additional_nodes)} additional nodes in Complete SI-C")
    
    # For each additional node, try adding it along with its edges to nodes in our current graph
    for node in additional_nodes:
        # Get all edges from this node to nodes in our current graph
        node_edges = [(node, neighbor) for neighbor in complete_sic.neighbors(node) 
                      if neighbor in common_graph.nodes()]
        
        if not node_edges:
            print(f"Node {node} has no connections to our current graph, skipping")
            continue
        
        # Add the node and its edges to our graph
        common_graph.add_node(node)
        temp_edges_added = 0
        
        for edge in node_edges:
            common_graph.add_edge(*edge)
            temp_edges_added += 1
        
        # Check if it's still a subgraph of CK-31
        GM = isomorphism.GraphMatcher(ck31_relabeled, common_graph)
        if GM.subgraph_is_monomorphic():
            print(f"Added node {node} with {temp_edges_added} edges, graph is still a subgraph of CK-31")
        else:
            # If not, remove the node and all its edges
            for edge in node_edges:
                if common_graph.has_edge(*edge):
                    common_graph.remove_edge(*edge)
            common_graph.remove_node(node)
            print(f"Adding node {node} would make the graph not a subgraph of CK-31, skipping")
    
    # Final check of our conditions
    GM1 = isomorphism.GraphMatcher(ck31_relabeled, common_graph)
    is_subgraph_of_ck31 = GM1.subgraph_is_monomorphic()
    
    GM2 = isomorphism.GraphMatcher(complete_sic, common_graph)
    is_subgraph_of_complete_sic = GM2.subgraph_is_monomorphic()
    
    GM3 = isomorphism.GraphMatcher(common_graph, minimal_sic)
    contains_minimal_sic = GM3.subgraph_is_monomorphic()
    
    print(f"Is our graph a subgraph of CK-31? {is_subgraph_of_ck31}")
    print(f"Is our graph a subgraph of Complete SI-C? {is_subgraph_of_complete_sic}")
    print(f"Does our graph contain Minimal SI-C? {contains_minimal_sic}")
    
    if is_subgraph_of_ck31 and is_subgraph_of_complete_sic and contains_minimal_sic:
        print(f"Found a maximal common subgraph with {common_graph.number_of_nodes()} nodes and {common_graph.number_of_edges()} edges")
        return common_graph
    else:
        print("Could not find a common subgraph that satisfies all conditions.")
        return None

def draw_graph(G, title):
    """Draw a graph with a title using an optimized layout"""
    plt.figure(figsize=(12, 10))
    
    # Try different layouts to find the best one for visualization
    # Kamada-Kawai layout often gives good results for graph structure visualization
    pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes and edges with better styling
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def draw_overlay_comparison(minimal_sic, common_graph, complete_sic):
    """
    Draw an overlay comparison of the three graphs to visualize their differences.
    
    Parameters:
        minimal_sic (nx.Graph): The minimal SI-C graph
        common_graph (nx.Graph): The common subgraph we found
        complete_sic (nx.Graph): The complete SI-C graph
    """
    plt.figure(figsize=(15, 10))
    
    # Create a custom layout that works well for all three graphs
    # Use the same node positions for all graphs for consistent comparison
    pos = nx.spring_layout(complete_sic, seed=42, k=0.5)
    
    # Draw the complete SI-C graph edges in light gray (background)
    nx.draw_networkx_edges(complete_sic, pos, alpha=0.2, width=1.0, edge_color='gray')
    
    # Draw the common graph edges in blue (middle layer)
    nx.draw_networkx_edges(common_graph, pos, alpha=0.6, width=2.0, edge_color='blue')
    
    # Draw the minimal SI-C graph edges in red (top layer)
    nx.draw_networkx_edges(minimal_sic, pos, alpha=0.8, width=1.5, edge_color='red')
    
    # Draw all nodes
    # First, draw all nodes from complete SI-C in light gray
    nx.draw_networkx_nodes(complete_sic, pos, node_color='lightgray', node_size=300, alpha=0.5)
    
    # Draw nodes from common graph in blue
    nx.draw_networkx_nodes(common_graph, pos, node_color='skyblue', node_size=400, alpha=0.7)
    
    # Draw nodes from minimal SI-C in red
    nx.draw_networkx_nodes(minimal_sic, pos, node_color='lightcoral', node_size=300, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(complete_sic, pos, font_size=10, font_weight='bold')
    
    # Add a legend
    plt.plot([0], [0], color='red', linewidth=1.5, label='Minimal SI-C')
    plt.plot([0], [0], color='blue', linewidth=2.0, label='Common Subgraph')
    plt.plot([0], [0], color='gray', linewidth=1.0, label='Complete SI-C')
    plt.legend(loc='upper right')
    
    plt.title("Comparison of Minimal SI-C, Common Subgraph, and Complete SI-C")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def draw_intermediate_vs_complete_comparison():
    """
    Draw a comparison between the Intermediate SI-C graph and the Complete SI-C graph
    to highlight which edges are in Complete SI-C but not in Intermediate SI-C.
    """
    # Create both graphs
    intermediate_sic = create_intermediate_sic_graph()
    complete_sic = create_complete_sic_graph()
    
    # Relabel intermediate_sic to have node labels starting from 1 to match complete_sic
    intermediate_sic = nx.convert_node_labels_to_integers(intermediate_sic, first_label=1)
    
    # Check if intermediate_sic is a subgraph of complete_sic
    GM = isomorphism.GraphMatcher(complete_sic, intermediate_sic)
    is_subgraph = GM.subgraph_is_monomorphic()
    
    if is_subgraph:
        print("Intermediate SI-C is confirmed to be a subgraph of Complete SI-C")
        # Get the mapping from intermediate_sic nodes to complete_sic nodes
        mapping = GM.mapping
        
        # Relabel intermediate_sic nodes to match their positions in complete_sic
        intermediate_sic_aligned = nx.relabel_nodes(intermediate_sic, mapping)
    else:
        print("Warning: Intermediate SI-C is not a subgraph of Complete SI-C")
        intermediate_sic_aligned = intermediate_sic
    
    # Get statistics
    intermediate_nodes = set(intermediate_sic_aligned.nodes())
    complete_nodes = set(complete_sic.nodes())
    
    intermediate_edges = set(intermediate_sic_aligned.edges())
    intermediate_edges.update([(v, u) for u, v in intermediate_edges])  # Add reverse edges for undirected comparison
    
    complete_edges = set(complete_sic.edges())
    
    # Find edges that are in complete but not in intermediate
    exclusive_to_complete = []
    for edge in complete_edges:
        if edge not in intermediate_edges and (edge[1], edge[0]) not in intermediate_edges:
            exclusive_to_complete.append(edge)
    
    # Find nodes that are in complete but not in intermediate
    exclusive_nodes = complete_nodes - intermediate_nodes
    
    print(f"Intermediate SI-C has {len(intermediate_nodes)} nodes and {len(intermediate_sic_aligned.edges())} edges")
    print(f"Complete SI-C has {len(complete_nodes)} nodes and {len(complete_sic.edges())} edges")
    print(f"There are {len(exclusive_to_complete)} edges exclusive to Complete SI-C")
    print(f"There are {len(exclusive_nodes)} nodes exclusive to Complete SI-C: {sorted(exclusive_nodes)}")
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Use Kamada-Kawai layout for better structure visualization
    pos = nx.kamada_kawai_layout(complete_sic)
    
    # Draw the complete SI-C edges that are not in intermediate in red (highlighted)
    nx.draw_networkx_edges(complete_sic, pos, edgelist=exclusive_to_complete, 
                          alpha=0.9, width=2.5, edge_color='red')
    
    # Draw the intermediate SI-C edges in blue
    nx.draw_networkx_edges(intermediate_sic_aligned, pos, alpha=0.7, width=1.5, edge_color='blue')
    
    # Draw nodes in intermediate SI-C
    nx.draw_networkx_nodes(intermediate_sic_aligned, pos, node_color='skyblue', 
                          node_size=500, alpha=0.8)
    
    # Draw nodes exclusive to complete SI-C
    exclusive_node_list = list(exclusive_nodes)
    if exclusive_node_list:
        nx.draw_networkx_nodes(complete_sic, pos, nodelist=exclusive_node_list, 
                              node_color='lightcoral', node_size=500, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(complete_sic, pos, font_size=12, font_weight='bold')
    
    # Add a legend
    plt.plot([0], [0], color='blue', linewidth=1.5, label='Edges in Intermediate SI-C')
    plt.plot([0], [0], color='red', linewidth=2.5, label='Edges exclusive to Complete SI-C')
    plt.legend(loc='upper right', fontsize=12)
    
    plt.title("Comparison of Intermediate SI-C and Complete SI-C", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def draw_minimal_vs_intermediate_comparison():
    """
    Draw a comparison between the Minimal SI-C graph and the Intermediate SI-C graph
    to highlight which edges and vertices exist only in Intermediate SI-C.
    """
    # Create both graphs
    minimal_sic = create_minimal_sic_graph()
    intermediate_sic = create_intermediate_sic_graph()
    
    # Relabel intermediate_sic to have node labels starting from 1 to match minimal_sic
    intermediate_sic = nx.convert_node_labels_to_integers(intermediate_sic, first_label=1)
    
    # Check if minimal_sic is a subgraph of intermediate_sic
    GM = isomorphism.GraphMatcher(intermediate_sic, minimal_sic)
    is_subgraph = GM.subgraph_is_monomorphic()
    
    if is_subgraph:
        print("Minimal SI-C is confirmed to be a subgraph of Intermediate SI-C")
        # Get the mapping from minimal_sic nodes to intermediate_sic nodes
        mapping = GM.mapping
        
        # Relabel minimal_sic nodes to match their positions in intermediate_sic
        minimal_sic_aligned = nx.relabel_nodes(minimal_sic, mapping)
    else:
        print("Warning: Minimal SI-C is not a subgraph of Intermediate SI-C")
        minimal_sic_aligned = minimal_sic
    
    # Get statistics
    minimal_nodes = set(minimal_sic_aligned.nodes())
    intermediate_nodes = set(intermediate_sic.nodes())
    
    minimal_edges = set(minimal_sic_aligned.edges())
    minimal_edges.update([(v, u) for u, v in minimal_edges])  # Add reverse edges for undirected comparison
    
    intermediate_edges = set(intermediate_sic.edges())
    
    # Find edges that are in intermediate but not in minimal
    exclusive_to_intermediate = []
    for edge in intermediate_edges:
        if edge not in minimal_edges and (edge[1], edge[0]) not in minimal_edges:
            exclusive_to_intermediate.append(edge)
    
    # Find nodes that are in intermediate but not in minimal
    exclusive_nodes = intermediate_nodes - minimal_nodes
    
    print(f"Minimal SI-C has {len(minimal_nodes)} nodes and {len(minimal_sic_aligned.edges())} edges")
    print(f"Intermediate SI-C has {len(intermediate_nodes)} nodes and {len(intermediate_sic.edges())} edges")
    print(f"There are {len(exclusive_to_intermediate)} edges exclusive to Intermediate SI-C")
    print(f"There are {len(exclusive_nodes)} nodes exclusive to Intermediate SI-C: {sorted(exclusive_nodes)}")
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Use Kamada-Kawai layout for better structure visualization
    pos = nx.kamada_kawai_layout(intermediate_sic)
    
    # Draw the minimal SI-C edges in blue
    nx.draw_networkx_edges(minimal_sic_aligned, pos, alpha=0.7, width=1.5, edge_color='blue')
    
    # Draw the intermediate SI-C edges that are not in minimal in green (highlighted)
    nx.draw_networkx_edges(intermediate_sic, pos, edgelist=exclusive_to_intermediate, 
                          alpha=0.9, width=2.5, edge_color='green')
    
    # Draw nodes in minimal SI-C
    nx.draw_networkx_nodes(minimal_sic_aligned, pos, node_color='skyblue', 
                          node_size=500, alpha=0.8)
    
    # Draw nodes exclusive to intermediate SI-C
    exclusive_node_list = list(exclusive_nodes)
    if exclusive_node_list:
        nx.draw_networkx_nodes(intermediate_sic, pos, nodelist=exclusive_node_list, 
                              node_color='lightgreen', node_size=500, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(intermediate_sic, pos, font_size=12, font_weight='bold')
    
    # Add a legend
    plt.plot([0], [0], color='blue', linewidth=1.5, label='Edges in Minimal SI-C')
    plt.plot([0], [0], color='green', linewidth=2.5, label='Edges exclusive to Intermediate SI-C')
    plt.legend(loc='upper right', fontsize=12)
    
    plt.title("Comparison of Minimal SI-C and Intermediate SI-C", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to find and display the common subgraph"""
    # Create and display all three graphs
    minimal_sic = create_minimal_sic_graph()
    intermediate_sic = create_intermediate_sic_graph()
    complete_sic = create_complete_sic_graph()
    ck31 = create_ck31_graph()
    
    draw_graph(minimal_sic, "Minimal SI-C Graph")
    draw_graph(intermediate_sic, "Intermediate SI-C Graph")
    draw_graph(complete_sic, "Complete SI-C Graph")
    draw_graph(ck31, "CK-31 Graph")
    
    # Compare minimal and intermediate SI-C
    draw_minimal_vs_intermediate_comparison()
    
    # Compare intermediate and complete SI-C
    draw_intermediate_vs_complete_comparison()
    
    # Find the common subgraph
    common_graph = find_common_subgraph()
    
    if common_graph:
        draw_graph(common_graph, "Common Subgraph")
        print(f"Found a common subgraph with {common_graph.number_of_nodes()} nodes and {common_graph.number_of_edges()} edges")
        print("Edges in the common subgraph:")
        for edge in sorted(common_graph.edges()):
            print(edge)
        
        # Print the graph6 representation of the common subgraph
        try:
            # First ensure node labels are integers starting from 0
            common_graph_relabeled = nx.convert_node_labels_to_integers(common_graph, first_label=0)
            graph6_bytes = nx.to_graph6_bytes(common_graph_relabeled, header=False)
            graph6_string = graph6_bytes.decode('ascii')
            print("\nGraph6 representation of the common subgraph:")
            print(graph6_string)
        except Exception as e:
            print(f"\nError generating graph6 representation: {str(e)}")
            print("This might be due to non-standard node labels or other graph properties.")
    else:
        print("Could not find a common subgraph with the simple approach.")
        print("A more sophisticated algorithm is needed.")

if __name__ == "__main__":
    main()