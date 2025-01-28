import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import pyvista as pv
import itertools
from itertools import combinations
from networkx.algorithms.planar_drawing import triangulate_embedding
import numpy as np
from yamada.sgd.sgd_modification import split_edges


def tutte_system(planar_graph):
    G, outer = triangulate_embedding(planar_graph, fully_triangulate=False)
    node_to_index = {node: index for index, node in enumerate(G.nodes())}
    n = G.number_of_nodes()
    assert n == planar_graph.number_of_nodes()
    inner_indices = np.array([node not in outer for node in G.nodes()])
    L = nx.laplacian_matrix(G).toarray()
    L = L[inner_indices, :]
    M = np.zeros((len(outer), n))
    for i, node in enumerate(outer):
        M[i, node_to_index[node]] = 1
    A = np.vstack([L, M])
    Z = np.zeros(A.shape)
    B = np.block([[A, Z], [Z, A]])
    # set locations of boundary nodes
    t = np.linspace(0, 2*np.pi, len(outer), endpoint=False)
    x, y = 100*np.cos(t), 100*np.sin(t)
    zeros = np.zeros(L.shape[0])
    b = np.hstack([zeros, x, zeros, y])
    return G, B, b


def tutte_embedding_positions(planar_graph):
    n = planar_graph.number_of_nodes()
    G, A, b = tutte_system(planar_graph)
    pos = np.linalg.solve(A, b)
    pos = np.rint(pos)
    x, y = pos[:n], pos[n:]
    ans = dict()
    for i, node in enumerate(G.nodes()):
        ans[node] = (int(x[i]), int(y[i]))
    return ans


def _position_spatial_graph_in_3d(G, z_height=20):
    def norm_label(X):
        L = X.label
        return repr(L) if not isinstance(L, str) else L
        
    P = G.planar_embedding()
    planar_pos = tutte_embedding_positions(P)
    system_node_pos = dict()
    for V in G.vertices:
        L = norm_label(V)
        x, y = planar_pos[V.label]
        system_node_pos[L] = (x, y, 0)

    other_node_pos = dict()
    for C in G.crossings:
        L = norm_label(C)
        x, y = planar_pos[C.label]
        other_node_pos[L + '+'] = (x, y, z_height)
        other_node_pos[L + '-'] = (x, y, -z_height)

    nodes_so_far = system_node_pos.copy()
    nodes_so_far.update(other_node_pos)

    def end_label(edge, i):
        i = i % 2
        X, x = edge.adjacent[i]
        L = norm_label(X)
        if X in G.crossings:
            L = L + '-' if x % 2 == 0 else L + '+'
        return L

    for E in G.edges:
        L = norm_label(E)
        x, y = planar_pos[E.label]
        A = end_label(E, 0)
        B = end_label(E, 1)
        z = (nodes_so_far[A][2] + nodes_so_far[B][2]) // 2
        other_node_pos[L] = (x, y, z)
        
    segments = list()
    vertex_inputs = set()
    for V in G.vertices:
        vertex_inputs.update((V, i) for i in range(V.degree))
    while len(vertex_inputs):
        V, i = vertex_inputs.pop()
        W, j = V.adjacent[i]
        one_seg = []
        while not W in G.vertices:
            if W in G.edges:
                L = norm_label(W)
                A = end_label(W, j)
                B = end_label(W, j + 1)
                one_seg += [A, L]
            W, j = W.flow(j)
        one_seg.append(norm_label(W))
        vertex_inputs.remove((W, j))
        segments.append(one_seg)

    return system_node_pos, other_node_pos, segments
        
def position_spatial_graph_in_3d(G, z_height=20):

    system_node_pos, other_node_pos, segments = _position_spatial_graph_in_3d(G, z_height)

    nodes = list(system_node_pos.keys())
    node_positions = list(system_node_pos.values())

    crossings = list(other_node_pos.keys())
    crossing_positions = list(other_node_pos.values())

    # Extract non crossings from crossings
    noncrossings, noncrossing_positions = zip(*[(crossing, crossing_position) for crossing, crossing_position in zip(crossings, crossing_positions) if "C" not in crossing])

    # Only extract non crossings
    # Merge nodes and crossings
    nodes.extend(noncrossings)
    node_positions.extend(noncrossing_positions)

    segments = split_edges(segments)

    # Convert segments from lists to tuples

    return nodes, node_positions, segments

def plot_spatial_graph_diagram():
    pass


def plot_spatial_graph(nodes, node_positions, edges, contiguous_sub_edges, contiguous_sub_edge_positions):

    # Define a list of colors to cycle through
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    color_cycle = itertools.cycle(color_list)

    # plotter = pv.Plotter()
    p = pv.Plotter(shape=(1, 2), window_size=[2000, 1000])

    # Plot the 3D Spatial Graph in the first subplot
    p.subplot(0, 0)
    p.add_title("3D Spatial Graph")

    # # Function to calculate offset position
    # def calculate_offset_position(position, offset=[0.1, 0.1, 0.1]):
    #     return position + np.array(offset)

    for contiguous_edge, contiguous_edge_positions_i in zip(contiguous_sub_edges, contiguous_sub_edge_positions):
        start_node = contiguous_edge[0]
        end_node = contiguous_edge[-1]
        start_position = contiguous_edge_positions_i[0][0]
        end_position = contiguous_edge_positions_i[-1][1]

        # TODO Use something more consistent than if "string" in node
        # color = "red" if "Crossing" in start_node else "black"
        # p.add_mesh(pv.Sphere(radius=0.05, center=start_position), color=color, opacity=0.5)
        # offset_start_position = calculate_offset_position(start_position)
        # p.add_point_labels([offset_start_position], [f"{start_node}"], point_size=0, font_size=12, text_color='black')

        # # TODO Use something more consistent than if "string" in node
        # color = "red" if "Crossing" in end_node else "black"
        # p.add_mesh(pv.Sphere(radius=0.05, center=end_position), color=color, opacity=0.5)
        # offset_end_position = calculate_offset_position(end_position)
        # p.add_point_labels([offset_end_position], [f"{end_node}"], point_size=0, font_size=12, text_color='black')

    # Plot the Projected lines
    for i, contiguous_sub_edge_positions_i in enumerate(contiguous_sub_edge_positions):
        lines = []
        color = next(color_cycle)
        for sub_edge_position_1, sub_edge_position_2 in contiguous_sub_edge_positions_i:
            start = sub_edge_position_1
            end = sub_edge_position_2

            line = pv.Line(start, end)
            lines.append(line)

        linear_spline = pv.MultiBlock(lines)
        # p.add_mesh(linear_spline, line_width=5, color=color)
        p.add_mesh(linear_spline, line_width=5, color="k")

    # Configure the plot
    p.view_isometric()
    p.show_axes()

    # Reset the color cycle for 2D edges
    color_cycle = itertools.cycle(color_list)

    # Plot the 2D Projection in the second subplot
    p.subplot(0, 1)
    p.add_title("2D Projection")

    # Plot the Projected lines in 2D
    for i, contiguous_sub_edge_positions_i in enumerate(contiguous_sub_edge_positions):
        lines = []
        color = next(color_cycle)
        for sub_edge_position_1, sub_edge_position_2 in contiguous_sub_edge_positions_i:
            start = sub_edge_position_1
            end = sub_edge_position_2

            line = pv.Line((start[0], 0, start[2]), (end[0], 0, end[2]))
            lines.append(line)

        linear_spline = pv.MultiBlock(lines)
        p.add_mesh(linear_spline, line_width=5, color=color, label=f"Edge {i}")

    # Configure the plot
    p.view_xz()
    p.show_axes()
    # p.add_legend(size=(0.1, 0.5), border=True, bcolor='white', loc='center right')

    return p


# def plot_spatial_graph(nodes, node_positions, edges):
#
#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot nodes
#     for node, position in node_positions.items():
#         ax.scatter(*position, label=node)
#
#     # Plot edges
#     for edge in edges:
#         edge_positions = [node_positions[node] for node in edge]
#         edge_positions = np.array(edge_positions)
#         ax.plot(edge_positions[:, 0], edge_positions[:, 1], edge_positions[:, 2])
#
#     plt.show()



# def add_curved_edge(ax, pos, n1, n2, over=True):
#     """Add a curved edge between two nodes."""
#     x1, y1 = pos[n1]
#     x2, y2 = pos[n2]
#     mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
#     curve_factor = 0.2
#     control_x = mid_x + curve_factor * (y2 - y1)
#     control_y = mid_y - curve_factor * (x2 - x1)
#
#     path = Path([(x1, y1), (control_x, control_y), (x2, y2)],
#                 [Path.MOVETO, Path.CURVE3, Path.CURVE3])
#     patch = PathPatch(path, edgecolor='blue' if over else 'orange', lw=2, zorder=2 if over else 1)
#     ax.add_patch(patch)

# def plot_spatial_graph(graph, pos):
#     """Plot the graph with crossings visualized."""
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_aspect('equal')
#
#     # Draw nodes
#     for node, (x, y) in pos.items():
#         ax.plot(x, y, 'o', color='black', zorder=3)
#         ax.text(x, y, str(node), fontsize=12, ha='center', va='center')
#
#     # Draw edges
#     for edge in graph.edges:
#         n1, n2 = edge
#         # Example: Assume alternating "over" and "under" crossings
#         over = graph.edges[edge].get('over', True)
#         add_curved_edge(ax, pos, n1, n2, over)
#
#     plt.axis('off')
#     plt.show()
#
# # Create a sample graph
# G = nx.Graph()
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A square graph
#
# # Add "over/under" attributes to crossings
# for i, edge in enumerate(G.edges):
#     G.edges[edge]['over'] = (i % 2 == 0)
#
# # Compute positions
# pos = nx.planar_layout(G)  # Use planar layout for simplicity
# plot_spatial_graph(G, pos)


# def plot(self, highlight_nodes=None, highlight_labels=None):
#     """
#     Plots the spatial graph diagram, labeling intermediate edges with index numbers
#     and intermediate nodes with full index assignments.
#     """
#
#     # Step 1: Create the planar-friendly graph
#     planar_graph, node_labels, edge_labels = self.planar_embedding()
#
#     # Step 2: Generate the planar embedding
#     is_planar, embedding = nx.check_planarity(planar_graph)
#     if not is_planar:
#         raise ValueError("The graph is not planar!")
#     pos = nx.planar_layout(embedding)
#
#     # Step 3: Separate node types
#     edges = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Edge"]
#     vertices = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Vertex"]
#     crossings = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Crossing"]
#     regular_nodes = [n for n, d in planar_graph.nodes(data=True) if d["type"] != "Intermediate"]
#     intermediate_nodes = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Intermediate"]
#
#     # Initialize the plot
#     plt.figure(figsize=(12, 12))
#
#     # Draw the edges
#     nx.draw_networkx_edges(planar_graph, pos)
#
#     # Label the edges
#     nx.draw_networkx_edge_labels(
#         planar_graph,
#         pos,
#         edge_labels=edge_labels,
#         font_size=8,
#         font_color="black",
#         rotate=False,
#     )
#
#     # Draw the nodes
#     nx.draw_networkx_nodes(
#         planar_graph,
#         pos,
#         nodelist=edges,
#         node_color="gray",
#         node_size=300,
#         alpha=0.7
#     )
#
#     nx.draw_networkx_nodes(
#         planar_graph,
#         pos,
#         nodelist=vertices,
#         node_color="lightgreen",
#         node_size=300,
#         alpha=0.7
#     )
#
#     nx.draw_networkx_nodes(
#         planar_graph,
#         pos,
#         nodelist=crossings,
#         node_color="lightblue",
#         node_size=300,
#         alpha=0.7
#     )
#
#     # Label the nodes
#     nx.draw_networkx_labels(
#         planar_graph,
#         pos,
#         labels={n: n for n in regular_nodes},
#         font_size=10,
#     )
#
#     if highlight_nodes:
#         nx.draw_networkx_nodes(
#             planar_graph,
#             pos,
#             nodelist=highlight_nodes,
#             node_color="yellow",
#             node_size=2000,
#             label="Highlighted Nodes",
#             alpha=0.4,
#             edgecolors="orange"
#         )
#
#     # State all object-index assignments
#     intermediate_label_text = "Object-Index Pairs \n" + "\n".join(f"{label}" for node, label in node_labels.items())
#     plt.gcf().text(
#         0.85, 0.5,  # Position the text box to the right of the plot
#         intermediate_label_text,
#         fontsize=10,
#         va="center",
#         bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.9),
#     )
#
#     # Step 10: Remove axis and spines
#     plt.axis("off")  # Turn off axes, ticks, and labels
#     ax = plt.gca()  # Get the current axis
#     for spine in ax.spines.values():
#         spine.set_visible(False)  # Hide all spines
#
#     # Show the plot
#     plt.title("Planar Embedding of the Spatial Graph Diagram")
#     plt.show()