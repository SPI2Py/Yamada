"""EXAMPLE

Note: This requires the plantri executable to be in the current working directory.
Plantri can be downloaded from http://users.cecs.anu.edu.au/~bdm/plantri/.
It is a C program, so you will need to compile it yourself. It is supported for Linux and Mac OS X.
"""


# %% Import Statements


import networkx as nx
import matplotlib.pyplot as plt

from yamada import enumerate_yamada_classes
from yamada.sgd.topological_distance import compute_min_distance

# %% Define the System Architecture and Component Geometries

# The system architecture is a NetworkX graph where the nodes represent components and the edges
# represent connections between components. The nodes are labeled with integers starting from 0.

# Currently, components must be either 2- or 3-valent. Please refer to the documentation for
# more information.

# User Input: System architecture
sa = [(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 0), (2, 5), (3, 5), (4, 5)]

# Create a networkx graph from the system architecture
sa_graph = nx.MultiGraph()
sa_graph.add_edges_from(sa)

# Plot the system architecture
# nx.draw(sa_graph, with_labels=True)
# plt.show()


# %% Enumerate all Unique Spatial Topologies

# User Input
number_of_crossings = 4

unique_spatial_topologies, number_topologies = enumerate_yamada_classes(sa_graph, number_of_crossings)

sgds = list(unique_spatial_topologies.values())

from itertools import combinations
sgd_pairs = list(combinations(sgds, 2))

print("Number of Enumerated Spatial Topologies: ", number_topologies)
print("Number of Unique Spatial Topologies: ", len(unique_spatial_topologies))
print("Number of Pairs of Unique Spatial Topologies: ", len(sgd_pairs))


# td1 = compute_min_distance(sgd_pairs[0][0], sgd_pairs[0][1])
# print("Topological Distance: ", td1)

# Loop through each SGD pair and construct a topological distance network
aggregated_network = nx.DiGraph()

# Process each pair of diagrams
for diagram1, diagram2 in sgd_pairs:
    # Compute the network for the current pair
    _, pair_network = compute_min_distance(diagram1, diagram2)

    # Add nodes and edges to the aggregated network
    for node, neighbors in pair_network.items():
        for neighbor in neighbors:
            aggregated_network.add_edge(node, neighbor)


# Visualize the aggregated network
import matplotlib.pyplot as plt

nx.draw(
    aggregated_network,
    with_labels=False,
    node_size=1000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold"
)
plt.show()
plt.savefig('graph3.png')

print("Done")

# %% Generate A Near-Planar Geometric Realization of Each Unique Spatial Topology


# sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)
#
# # Convert each near-planar geometric realization into a SpatialGraph object
# spatial_graphs = []
# for sg_input in sg_inputs:
#     sg = SpatialGraph(*sg_input)
#     spatial_graphs.append(sg)
#     sg.plot()
#
#     sgd = sg.create_spatial_graph_diagram()
#     print("Yamada Polynomial: ", sgd.normalized_yamada_polynomial())

