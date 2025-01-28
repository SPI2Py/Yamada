from yamada import SpatialGraph
from yamada.sgd.reidemeister import *

# from example_routing import node_labels, node_positions, edges
from example_cube import node_labels, node_positions, edges
# from example_cube_2 import node_labels, node_positions, edges
# from example_cube_3 import node_labels, node_positions, edges
# from example_cube_4 import node_labels, node_positions, edges
# from example_cube_5 import node_labels, node_positions, edges

# Set the random seed for reproducibility
# random.seed(0)
# np.random.seed(0)

# Instantiate the SpatialGraph object
sg = SpatialGraph(nodes=node_labels,
                  node_positions=node_positions,
                  edges=edges)

# Plot
sg.plot()

# # Create the spatial graph diagram (necessary for calculating the Yamada polynomial)
# sgd = sg.create_spatial_graph_diagram()
#
# print(f"Crossings: {len(sgd.crossings)}")
#
# # ...
# n_tries = 15
# sgd, r1_count, r2_count, r3_count = reidemeister_simplify(sgd, n_tries=n_tries)
# print(f"R1: {r1_count}, R2: {r2_count}, R3: {r3_count}, Remaining Crossings: {len(sgd.crossings)}")
#
# # if len(sgd.crossings) <= 10:
# #     yp = sgd.normalized_yamada_polynomial()
# #     print(yp)






