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


        
def position_spatial_graph_in_3d(G, z_height=20):

    def norm_label(X):
        L = X.label
        return repr(L) if not isinstance(L, str) else L

    def end_label(edge, i):
        i = i % 2
        X, x = edge.adjacent[i]
        L = norm_label(X)
        if X in G.crossings:
            L = L + '-' if x % 2 == 0 else L + '+'
        return L

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

    ###

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

def plot_spatial_graph_diagram(sgd):
    """
    Plots the spatial graph diagram in 3D using PyVista.
    Labels intermediate edges with index numbers and intermediate nodes with full index assignments.
    """
    # , planar_graph, node_labels, edge_labels
    # # Check for planarity
    # is_planar, embedding = nx.check_planarity(planar_graph)
    # if not is_planar:
    #     raise ValueError("The graph is not planar!")

    # Generate 2D positions for the planar embedding
    # pos = nx.planar_layout(embedding)
    nodes, node_positions, edges = position_spatial_graph_in_3d(sgd)

    # Extend positions to 3D by adding a z-coordinate (all zero for planar layout)
    # pos_3d = {node: np.array([x, y, 0]) for node, (x, y) in pos.items()}

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Plot nodes as spheres
    for node, coords in zip(nodes, node_positions):
        sphere = pv.Sphere(radius=0.5, center=coords)
        plotter.add_mesh(sphere, color="black", opacity=0.8, label=str(node))

    # Plot edges as tubes
    pos_dict = dict(zip(nodes, node_positions))
    for edge in edges:
        line = pv.Line(pos_dict[edge[0]], pos_dict[edge[-1]])
        plotter.add_mesh(line.tube(radius=0.1), color="black", label=str(edge))


    # # Add nodes as spheres
    # for node, coords in pos_3d.items():
    #
    #     if planar_graph.nodes[node]["type"] == "Edge":
    #         color = "black"
    #         opacity = 0.8
    #         sphere = pv.Sphere(radius=0.01, center=coords)
    #     elif planar_graph.nodes[node]["type"] == "Intermediate":
    #         color = "black"
    #         opacity = 0.8
    #         sphere = pv.Sphere(radius=0.01, center=coords)
    #     elif planar_graph.nodes[node]["type"] == "Vertex":
    #         color = "lightblue"
    #         opacity = 0.8
    #         sphere = pv.Sphere(radius=0.05, center=coords)
    #     elif planar_graph.nodes[node]["type"] == "Crossing":
    #         color = "green"
    #         opacity = 0.8
    #         sphere = pv.Sphere(radius=0.05, center=coords)
    #     else:
    #         raise ValueError("Unknown node type!")
    #         # color = "orange"
    #         # opacity = 0.5
    #         # sphere = pv.Sphere(radius=0.075, center=coords)
    #
    #     plotter.add_mesh(sphere, color=color, opacity=opacity, label=str(node))
    #
    # # Add edges as tubes
    # for edge, edge_label in zip(planar_graph.edges, edge_labels.values()):
    #     start, end = edge
    #     line = pv.Line(pos_3d[start], pos_3d[end])
    #     plotter.add_mesh(line.tube(radius=0.01), color="black", label=str(edge_label))
    #
    # # Add node labels
    # for node, coords in pos_3d.items():
    #     label = node_labels.get(node, str(node))
    #     plotter.add_point_labels(
    #         [coords],
    #         [label],
    #         point_size=10,
    #         font_size=12,
    #         bold=True,
    #         text_color="black",
    #     )
    #
    # # Add edge labels
    # for edge, edge_label in zip(planar_graph.edges, edge_labels.values()):
    #     midpoint = (pos_3d[edge[0]] + pos_3d[edge[1]]) / 2
    #     plotter.add_point_labels(
    #         [midpoint],
    #         [str(edge_label)],
    #         point_size=10,
    #         font_size=10,
    #         bold=False,
    #         text_color="blue",
    #     )
    #
    # # State all object-index assignments
    # intermediate_label_text = "Object-Index Pairs \n" + "\n".join(f"{label}" for node, label in node_labels.items())
    # text_coords = [0.8, 0.2, 0.0]  # Position the text box in normalized coordinates
    # plotter.add_text(
    #     intermediate_label_text,
    #     position="upper_right",
    #     font_size=10,
    #     color="black",
    #     viewport=True
    # )
    #
    # # Finalize the plot
    # plotter.add_axes()
    # plotter.add_legend()
    # plotter.show(title="Spatial Graph Diagram")
    return plotter

def plot_spatial_graph(nodes, node_positions, edges, contiguous_sub_edges, contiguous_sub_edge_positions):

    # Define a list of colors to cycle through
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    color_cycle = itertools.cycle(color_list)

    # plotter = pv.Plotter()
    p = pv.Plotter(shape=(1, 2), window_size=[2000, 1000])

    # Plot the 3D Spatial Graph in the first subplot
    p.subplot(0, 0)
    p.add_title("3D Spatial Graph")


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
    # p.view_isometric()
    # p.view_xz()
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

    # Link the two plots
    p.link_views()

    return p
