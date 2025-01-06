"""EXAMPLE

Note: This requires the plantri executable to be in the current working directory.
Plantri can be downloaded from http://users.cecs.anu.edu.au/~bdm/plantri/.
It is a C program, so you will need to compile it yourself. It is supported for Linux and Mac OS X.
"""


# %% Import Statements


import networkx as nx
import matplotlib.pyplot as plt

from yamada import enumerate_yamada_classes

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
nx.draw(sa_graph, with_labels=True)
plt.show()


# %% Enumerate all Unique Spatial Topologies

# User Input
number_of_crossings = 2

unique_spatial_topologies, number_topologies = enumerate_yamada_classes(sa_graph, number_of_crossings)


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

