import networkx as nx
import matplotlib.pyplot as plt

# Define adjacency list
adj_list = {
    1: [2,3,4,5],
    2: [1,17,21,3],
    3: [2,25,18,1],
    4: [1,5,6,8],
    5: [1,4,7,9],
    6: [4,10],
    7: [5,11],
    8: [4,15],
    9: [5,23],
    10: [6,12,25],
    11: [7,14,13],
    12: [10,17],
    13: [11,18],
    14: [11,17],
    15: [8,16,19],
    16: [15,18],
    17: [12,14,21,2],
    18: [13,16,25,3],
    19: [15,21],
    20: [10, 25],
    21: [17,19,22,2],
    22: [21,23],
    24: [23,25],
    25: [10,18,24,3]
}

# Create graph
G = nx.Graph()

# Add edges from adjacency list
for node, neighbors in adj_list.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Print edges
print(list(G.edges))

# Get sorted list of nodes
nodes = sorted(G.nodes())
n = len(nodes)

# Print upper triangle as a continuous string (excluding diagonals)
print("\nUpper triangle of adjacency matrix as a string (excluding diagonals):")
result = ""
for j in range(n):
    for i in range(j):
        if G.has_edge(nodes[i], nodes[j]):
            result += "1"
        else:
            result += "0"
print(result)

# Plot graph
"""nx.draw(G, with_labels=True)
plt.show()
"""

#RCL form of order 25 subgraph: 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000001100000000001100010000000010101000000000110100000000000010100000100000000110000010000000000100100000000100000111000000000000000000010100000000010100000000000000000011100000100100000010000100000100100

