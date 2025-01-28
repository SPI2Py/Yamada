"""Spatial Graphs

This module contains classes and functions for working with spatial graphs.
"""


import numpy as np

import itertools
from itertools import combinations
import pyvista as pv
import matplotlib.colors as mcolors
from scipy.stats import qmc

from ..sg.geometry import (rotate,
                                compute_line_segment_intersection,
                                compute_intermediate_y_position,
                                compute_3D_intersection,
                                compute_counter_clockwise_angle)

from ..sgd.diagram_elements import Vertex, Crossing
from ..sgd.spatial_graph_diagrams import SpatialGraphDiagram
from ..utils.visualization import plot_spatial_graph

class SpatialGraph:
    """
    A class to represent a spatial graph.
    """

    def __init__(self,
                 nodes: list[str],
                 node_positions: dict,
                 edges: list[tuple[str, str]],
                 rotation=None):

        self.nodes = nodes
        self.edges = edges
        self.edge_pairs = list(combinations(self.edges, 2))

        self.adjacent_edge_pairs = self._get_adjacent_edge_pairs()
        self.nonadjacent_edge_pairs = [edge_pair for edge_pair in self.edge_pairs if
                                       edge_pair not in self.adjacent_edge_pairs]

        # Temporarily convert node positions to array
        node_positions_list = [node_positions[node] for node in nodes]
        node_positions = np.array(node_positions_list)

        # Project the spatial graph onto a random the xz-plane
        self.node_positions_3d = self.project(node_positions, forced_rotation=rotation)
        self.node_positions_dict_3d = {node: position for node, position in zip(nodes, node_positions)}

        self.node_positions_2d = self.node_positions_3d[:, [0, 2]]
        self.node_positions_dict_2d = {node: position for node, position in zip(nodes, self.node_positions_2d)}

        self.crossings, self.crossing_positions, self.crossing_edge_pairs = self.get_crossings()


    def get_adjacent_nodes(self, reference_node: str) -> list[str]:
        adjacent_nodes = []

        for edge in self.edges:
            if reference_node == edge[0] or reference_node == edge[1]:
                adjacent_nodes += [node for node in edge if node != reference_node]

        return adjacent_nodes


    def _get_adjacent_edge_pairs(self):

        adjacent_edge_pairs = set()  # Using a set to avoid duplicates

        # Create a dictionary that maps each node to the edges that contain it
        node_to_edges = {}

        for edge in self.edges:
            a, b = edge
            if a not in node_to_edges:
                node_to_edges[a] = []
            if b not in node_to_edges:
                node_to_edges[b] = []
            node_to_edges[a].append(edge)
            node_to_edges[b].append(edge)

        # Loop through each node and find pairs of edges connected to that node
        for node, edges in node_to_edges.items():
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    # Add the pair of edges to the set (unordered pair)
                    adjacent_edge_pairs.add((edges[i], edges[j]))

        return list(adjacent_edge_pairs)  # Convert set to list for the result


    def get_vertices_and_crossings_of_edge(self, reference_edge: tuple[str, str]):
        """
        Returns the vertices and crossings of an edge, ordered from left to right.
        """

        # Get edge nodes and positions
        edge_nodes = [node for node in reference_edge]
        edge_node_positions = [self.node_positions_dict_2d[node] for node in edge_nodes]

        # Get crossing and positions (if applicable)
        edge_crossings = []
        edge_crossing_positions = []

        for edge_pair, position, crossing in zip(self.crossing_edge_pairs, self.crossing_positions, self.crossings):
            if reference_edge in edge_pair:
                edge_crossings.append(crossing)
                edge_crossing_positions.append(position)

        if len(edge_crossings) > 0:
            # Merge the vertices and crossings
            adjacent_nodes = edge_nodes + edge_crossings
            adjacent_positions = np.vstack((edge_node_positions, edge_crossing_positions))

        else:
            adjacent_nodes = edge_nodes
            adjacent_positions = edge_node_positions

        # Order vertices and crossings from left to right by x position (i.e., ascending position index 0)
        ordered_nodes = [node for _, node in
                         sorted(zip(adjacent_positions, adjacent_nodes), key=lambda pair: pair[0][0])]

        return ordered_nodes


    def get_sub_edges(self):
        """
        Sub-edges constitute:
        1. Edges that are not incident of a crossing (i.e., the edge is the sub-edge)
        2. When an edge is incident to a crossing(s), the edge is divided into one or more sub-edges depending on the
        number of crossings that the edge is incident to.
        """

        # Get the nodes and crossings of each edge, ordered from left to right
        edge_nodes_and_crossings = {}
        for edge in self.edges:
            edge_nodes_and_crossings[edge] = self.get_vertices_and_crossings_of_edge(edge)

        # Subdivide each edge if necessary.
        # If there are 2 nodes then there is one sub-edge, 3 nodes means 2 sub-edges, etc.

        sub_edges = []

        for edge, nodes_and_crossings in edge_nodes_and_crossings.items():

            if len(nodes_and_crossings) == 2:
                sub_edges.append(edge)

            else:
                sub_edges += [(nodes_and_crossings[i], nodes_and_crossings[i + 1]) for i in
                              range(len(nodes_and_crossings) - 1)]

        return sub_edges

    def get_contiguous_edges(self):

        # Get the sub-edges and their positions
        sub_edges, sub_edge_positions = self.subdivide_edges_with_crossings()

        # Get the nodes and crossings
        nodes = []
        for node_a, node_b in sub_edges:
            if node_a not in nodes:
                nodes.append(node_a)
            if node_b not in nodes:
                nodes.append(node_b)

        nodes_dict = {node: {'valency': 0, 'adjacent': []} for node in nodes}
        for sub_edge in sub_edges:
            a, b = sub_edge
            if a in nodes:
                nodes_dict[a]['valency'] += 1
                nodes_dict[a]['adjacent'].append(b)
            if b in nodes:
                nodes_dict[b]['valency'] += 1
                nodes_dict[b]['adjacent'].append(a)

        two_valent_nodes = [node for node in nodes if nodes_dict[node]['valency'] == 2]
        other_nodes_and_crossings = [node for node in nodes if node not in two_valent_nodes]

        # Start with a non-two-valent node or crossing, and then accumulate the sub-edges until another
        # non-two-valent node or crossing is reached. These are contiguous sub-edges.
        entry_points = []
        for node in other_nodes_and_crossings:
            node_entry_points = [(node, i) for i in range(nodes_dict[node]['valency'])]
            entry_points += node_entry_points

        unvisited_entry_points = entry_points.copy()

        max_iter = 1000
        iter_count = 0
        contiguous_sub_edges = []
        contiguous_sub_edge_positions = []
        while len(unvisited_entry_points) > 0:

            contiguous_sub_edge = []

            # Entry point
            entry_point = unvisited_entry_points[0]
            entry_node, entry_node_index = entry_point
            unvisited_entry_points.remove(entry_point)
            contiguous_sub_edge.append(entry_node)

            # Intermediate points
            previous_node = entry_node
            current_node = nodes_dict[entry_node]['adjacent'][entry_node_index]
            contiguous_sub_edge.append(current_node)

            while nodes_dict[current_node]['valency'] == 2:
                if nodes_dict[current_node]['adjacent'][0] == previous_node:
                    previous_node = current_node
                    current_node = nodes_dict[previous_node]['adjacent'][1]
                    contiguous_sub_edge.append(current_node)
                elif nodes_dict[current_node]['adjacent'][1] == previous_node:
                    previous_node = current_node
                    current_node = nodes_dict[previous_node]['adjacent'][0]
                    contiguous_sub_edge.append(current_node)
                else:
                    raise ValueError("Error in adjacent nodes.")

            # Exit point
            exit_node = current_node
            exit_node_index = nodes_dict[exit_node]['adjacent'].index(previous_node)
            unvisited_entry_points.remove((exit_node, exit_node_index))

            contiguous_sub_edges.append(contiguous_sub_edge)

            iter_count += 1
            if iter_count > max_iter:
                raise ValueError("Max iteration count reached.")

        # Now we will have the contiguous sub-edges, we can get their positions
        for contiguous_sub_edge in contiguous_sub_edges:
            contiguous_sub_edge_positions_i = []
            for i in range(len(contiguous_sub_edge) - 1):
                node_a = contiguous_sub_edge[i]
                node_b = contiguous_sub_edge[i + 1]
                edge = [node_a, node_b]
                edge_reversed = [node_b, node_a]
                if edge in sub_edges:
                    edge_index = sub_edges.index(edge)
                    contiguous_sub_edge_positions_i.append(sub_edge_positions[edge_index])
                elif edge_reversed in sub_edges:
                    edge_index = sub_edges.index(edge_reversed)
                    positions = sub_edge_positions[edge_index]
                    positions_reversed = [positions[1], positions[0]]
                    contiguous_sub_edge_positions_i.append(positions_reversed)
                else:
                    raise ValueError("Edge not found.")

            contiguous_sub_edge_positions.append(contiguous_sub_edge_positions_i)

        return contiguous_sub_edges,  contiguous_sub_edge_positions


    def get_edge_vertices_and_or_crossings(self, edge):
        """
        Returns a list of the vertices and crossings along a give edge, specifically ordered from -x to +x.
        FIXME What if the edge is perfectly vertical?
        """

        # Get edge nodes and positions
        edge_nodes = [node for node in edge]
        edge_node_positions = [self.node_positions_dict_2d[node] for node in edge_nodes]

        # Get crossing and positions (if applicable)
        edge_crossings = []
        edge_crossing_positions = []
        for edge_pair, position, crossing in zip(self.crossing_edge_pairs, self.crossing_positions, self.crossings):
            if edge in edge_pair:
                edge_crossings.append(crossing)
                edge_crossing_positions.append(position)

        if len(edge_crossings) > 0:
            # Merge the vertices and crossings
            adjacent_nodes = edge_nodes + edge_crossings
            adjacent_positions = np.vstack((edge_node_positions, edge_crossing_positions))

        else:
            adjacent_nodes = edge_nodes
            adjacent_positions = edge_node_positions

        # Order vertices and crossings from left to right by x position (i.e., ascending position index 0)
        ordered_nodes = [node for _, node in
                         sorted(zip(adjacent_positions, adjacent_nodes), key=lambda pair: pair[0][0])]

        return ordered_nodes

    def get_adjacent_nodes_or_crossings(self, reference_node):
        """
        Get the adjacent nodes to a given node.
        """

        adjacent_nodes = []

        for edge in self.get_sub_edges():
            if reference_node == edge[0] or reference_node == edge[1]:
                adjacent_nodes += [node for node in edge if node != reference_node]

        return adjacent_nodes

    def get_adjacent_nodes_projected_positions(self, reference_node: str):
        """
        Get the adjacent nodes to a given node.
        """
        adjacent_nodes = self.get_adjacent_nodes(reference_node)

        node_indices = [self.nodes.index(node) for node in adjacent_nodes]

        return self.node_positions_2d[node_indices]

    def cyclic_order_vertex(self,
                            reference_node:     str,
                            node_ordering_dict: dict = None) -> dict:
        """
        Get the cyclical order of adjacent vertexes and crossings for a given vertex.

        The spatial-topological calculations presume that nodes are ordered in a CCW rotation. While it does not matter
        which node is used as index zero, it is important that the node order is consistent.

        :param reference_node: The node to use as the reference node.
        :param node_ordering_dict: A dictionary of the node order.
        :return: A dictionary of the node order.
        """

        # If no node ordering dictionary is provided, use the default node ordering dictionary
        if node_ordering_dict is None:
            node_ordering_dict = {}

        # Get the projected node positions
        reference_node_position = self.node_positions_dict_2d[reference_node]

        # Initialize lists to store the adjacent node and edge information
        # Crossings are not relevant for this calculation since they exist along edges
        adjacent_nodes = self.get_adjacent_nodes_or_crossings(reference_node)
        adjacent_node_positions = self.get_adjacent_nodes_projected_positions(reference_node)

        # Shift nodes to the origin
        shifted_adjacent_node_positions = adjacent_node_positions - reference_node_position

        # Horizontal line (reference for angles)
        reference_vector = np.array([1, 0])

        rotations = []
        for shifted_adjacent_node_position in shifted_adjacent_node_positions:
            rotations.append(compute_counter_clockwise_angle(reference_vector, shifted_adjacent_node_position))

        ordered_nodes = [node for _, node in sorted(zip(rotations, adjacent_nodes))]

        ccw_node_ordering = {}
        for node in ordered_nodes:
            ccw_node_ordering[node] = ordered_nodes.index(node)

        node_ordering_dict[reference_node] = ccw_node_ordering

        return node_ordering_dict

    def cyclic_order_vertices(self):
        node_ordering_dict = {}
        for node in self.nodes:
            node_ordering_dict = self.cyclic_order_vertex(node, node_ordering_dict)
        return node_ordering_dict

    def subdivide_edges_with_crossings(self):

        # Initialize the values
        edges = self.edges
        nodes = self.nodes
        node_positions = self.node_positions_3d
        node_positions_dict = {node: position for node, position in zip(nodes, node_positions)}

        crossings, crossing_positions_2D, crossing_positions_3D, crossing_edge_pairs, crossing_positions_3D_dict = self.get_crossings_3D()

        # Get the nodes and crossings
        edge_nodes_and_or_crossings = []
        for edge in edges:
            edge_nodes_and_or_crossings_i = self.get_edge_vertices_and_or_crossings(edge)
            edge_nodes_and_or_crossings.append(edge_nodes_and_or_crossings_i)

        # Now get their positions
        edge_node_and_or_crossing_positions = []
        for edge in edge_nodes_and_or_crossings:
            edge_positions = []
            for vertex in edge:
                if "crossing" in vertex:
                    edge_positions.append(crossing_positions_3D_dict[vertex][edge[0]])
                else:
                    edge_positions.append(node_positions_dict[vertex])
            edge_node_and_or_crossing_positions.append(edge_positions)

        # Flatten the list of nodes and positions into lists of two
        subdivided_edges = []
        subdivided_edge_positions = []
        for edge_nodes_i, edge_node_positions_i in zip(edge_nodes_and_or_crossings, edge_node_and_or_crossing_positions):

            if len(edge_nodes_i) == 2:
                subdivided_edges.append(edge_nodes_i)
                subdivided_edge_positions.append(edge_node_positions_i)
            else:
                for i in range(len(edge_nodes_i) - 1):
                    subdivided_edges.append([edge_nodes_i[i], edge_nodes_i[i + 1]])
                    subdivided_edge_positions.append([edge_node_positions_i[i], edge_node_positions_i[i + 1]])


        return subdivided_edges, subdivided_edge_positions



    def get_node_or_crossing_projected_position(self, reference_node: str) -> np.ndarray:

        if reference_node in self.nodes:
            return self.node_positions_dict_2d[reference_node]

        elif reference_node in self.crossings:
            return self.crossing_positions[self.crossings.index(reference_node)]

        else:
            raise ValueError("Node or crossing not found.")


    def cyclic_order_crossing(self,
                              crossing:           str,
                              node_ordering_dict: dict = None):
        """
        Get the order of the overlapping nodes in the two edges. First is under, second is over.

        Use rotated node positions since rotation and project may change the overlap order.

        :param crossing: The crossing
        :param node_ordering_dict: A dictionary of node ordering for each node
        :return: overlap_order: The order of the overlapping nodes in edge 1 and edge 2
        """

        # If no node ordering dictionary is provided, use the default node ordering dictionary
        if node_ordering_dict is None:
            node_ordering_dict = {}

        # Get the edges that are connected to the crossing
        # Assumption: get_edges_of_crossing returns the edges in the order of the nodes
        edge_1, edge_2 = self.crossing_edge_pairs[self.crossings.index(crossing)]
        edge_1_nodes_and_crossings = self.get_edge_vertices_and_or_crossings(edge_1)
        edge_2_nodes_and_crossings = self.get_edge_vertices_and_or_crossings(edge_2)

        # Get the nodes that are adjacent to the crossing (whether a vertex or another crossing)
        # Assumption: The crossing is not the first or last node of the edge
        crossing_index_edge_1 = edge_1_nodes_and_crossings.index(crossing)
        crossing_index_edge_2 = edge_2_nodes_and_crossings.index(crossing)

        edge_1_left_node  = edge_1_nodes_and_crossings[crossing_index_edge_1 - 1]
        edge_1_right_node = edge_1_nodes_and_crossings[crossing_index_edge_1 + 1]
        edge_2_left_node  = edge_2_nodes_and_crossings[crossing_index_edge_2 - 1]
        edge_2_right_node = edge_2_nodes_and_crossings[crossing_index_edge_2 + 1]


        # Get the vertices that the beginning and end of each edge
        # While the crossing might be adjoined be other crossings, crossings only occur in 2D.
        # The 3D positions of the vertices are required to determine which edge is in front of the other.

        # Determine which vertex is the left and right vertex of each edge.
        # Assumption: Edges are ordered nodes left to right

        edge_1_left_vertex  = edge_1_nodes_and_crossings[0]
        edge_1_right_vertex = edge_1_nodes_and_crossings[-1]
        edge_2_left_vertex  = edge_2_nodes_and_crossings[0]
        edge_2_right_vertex = edge_2_nodes_and_crossings[-1]

        edge_1_left_vertex_position  = self.node_positions_3d[self.nodes.index(edge_1_left_vertex)]
        edge_1_right_vertex_position = self.node_positions_3d[self.nodes.index(edge_1_right_vertex)]
        edge_2_left_vertex_position  = self.node_positions_3d[self.nodes.index(edge_2_left_vertex)]
        edge_2_right_vertex_position = self.node_positions_3d[self.nodes.index(edge_2_right_vertex)]

        crossing_position = self.crossing_positions[self.crossings.index(crossing)]

        y_crossing_edge_1 = compute_intermediate_y_position(edge_1_left_vertex_position,
                                                            edge_1_right_vertex_position,
                                                            crossing_position[0],
                                                            crossing_position[1])

        y_crossing_edge_2 = compute_intermediate_y_position(edge_2_left_vertex_position,
                                                            edge_2_right_vertex_position,
                                                            crossing_position[0],
                                                            crossing_position[1])

        # Get the projected node positions
        reference_node_position = self.get_node_or_crossing_projected_position(crossing)

        # Initialize lists to store the adjacent node and edge information
        # Crossings are not relevant for this calculation since they exist along edges

        edge_1_left_vertex_position_projected  = self.get_node_or_crossing_projected_position(edge_1_left_vertex)
        edge_1_right_vertex_position_projected = self.get_node_or_crossing_projected_position(edge_1_right_vertex)
        edge_2_left_vertex_position_projected  = self.get_node_or_crossing_projected_position(edge_2_left_vertex)
        edge_2_right_vertex_position_projected = self.get_node_or_crossing_projected_position(edge_2_right_vertex)

        adjacent_vertex_positions_projected = [edge_1_left_vertex_position_projected, edge_1_right_vertex_position_projected, edge_2_left_vertex_position_projected, edge_2_right_vertex_position_projected]

        adjacent_nodes    = [edge_1_left_node, edge_1_right_node, edge_2_left_node, edge_2_right_node]
        adjacent_vertices = [edge_1_left_vertex, edge_1_right_vertex, edge_2_left_vertex, edge_2_right_vertex]

        # Shift nodes to the origin
        shifted_adjacent_vertex_positions = adjacent_vertex_positions_projected - reference_node_position

        # Horizontal line (reference for angles)
        reference_vector = np.array([1, 0])

        rotations = []

        # Convention: +y goes into the screen and -y goes out of the screen
        # Therefore the edge that is in front of the other is the edge that has the lower y value
        if y_crossing_edge_2 < y_crossing_edge_1:
            under_edge = edge_1
            over_edge  = edge_2
        elif y_crossing_edge_2 > y_crossing_edge_1:
            under_edge = edge_2
            over_edge  = edge_1
        else:
            raise Exception('These crossings should not intersect in 3D')

        # Calculate the angle of each node
        for shifted_adjacent_node_position in shifted_adjacent_vertex_positions:
            rotations.append(compute_counter_clockwise_angle(reference_vector, shifted_adjacent_node_position))

        # Sort the nodes by their angle
        ordered_nodes = [node for _, node in sorted(zip(rotations, adjacent_nodes))]

        # Sort the vertices by their angle (used for reference)
        ordered_vertices = [node for _, node in sorted(zip(rotations, adjacent_vertices))]

        # Check if the first node after the reference vector is on the under edge or the over edge
        first_vertex = ordered_vertices[0]
        if first_vertex in under_edge:
            crossing_indices = [0, 1, 2, 3]
        elif first_vertex in over_edge:
            crossing_indices = [1, 2, 3, 0]
        else:
            raise Exception("First node is not on either edge")

        # Assign the cyclic ordering
        ccw_crossing_ordering = {}
        for node, index in zip(ordered_nodes, crossing_indices):
            ccw_crossing_ordering[node] = index

        node_ordering_dict[crossing] = ccw_crossing_ordering

        return node_ordering_dict

    def cyclic_order_crossings(self):
        """
        Get the node ordering of the graph with crossings.

        TODO Replace crossing indices with crossings
        """
        # Create a dictionary that contains the cyclical ordering of every crossing
        crossing_ordering_dict = {}
        for crossing in self.crossings:
            crossing_ordering_dict = self.cyclic_order_crossing(crossing, crossing_ordering_dict)
        return crossing_ordering_dict

    def get_crossing_edge_order(self,
                                edge_1,
                                edge_2,
                                crossing_position):
        """
        Get the order of the overlapping nodes in the two edges. First is under, second is over.

        Use rotated node positions since rotation and project may change the overlap order.

        :param edge_1: Edge 1
        :param edge_2: Edge 2
        :param crossing_position: The point of intersection between the projections of edge 1 and edge 2

        :return: overlap_order: The order of the overlapping nodes in edge 1 and edge 2
        """

        overlap_order = []

        node_1, node_2 = edge_1
        node_3, node_4 = edge_2

        node_1_index = self.nodes.index(node_1)
        node_2_index = self.nodes.index(node_2)
        node_3_index = self.nodes.index(node_3)
        node_4_index = self.nodes.index(node_4)

        node_1_position = self.node_positions_3d[node_1_index]
        node_2_position = self.node_positions_3d[node_2_index]
        node_3_position = self.node_positions_3d[node_3_index]
        node_4_position = self.node_positions_3d[node_4_index]

        x_crossing, z_crossing = crossing_position

        y_crossing_1 = compute_intermediate_y_position(node_1_position, node_2_position, x_crossing, z_crossing)
        y_crossing_2 = compute_intermediate_y_position(node_3_position, node_4_position, x_crossing, z_crossing)

        # todo Only check if crossing is in bounds!
        if y_crossing_1 == y_crossing_2:
            raise RuntimeError('The edges are planar and therefore the interconnects they represent physically '
                               'intersect. This is not a valid spatial graph. Edges: {}, {}.'.format(edge_1, edge_2))

        elif y_crossing_1 > y_crossing_2:
            overlap_order.append(edge_2)
            overlap_order.append(edge_1)

        elif y_crossing_1 < y_crossing_2:
            overlap_order.append(edge_1)
            overlap_order.append(edge_2)

        else:
            raise NotImplementedError('There should be no else case.')

        # Convert list to tuple for hashing later on...
        overlap_order = tuple(overlap_order)

        return overlap_order


    def get_crossings(self):

        crossing_num = 0
        crossings    = []
        crossing_edge_pairs = []
        crossing_positions = []



        # TODO Modify nonadjacent edge pairs that are within some axis aligned bounding box

        for line_1, line_2 in self.nonadjacent_edge_pairs:

            a = self.node_positions_dict_2d[line_1[0]]
            b = self.node_positions_dict_2d[line_1[1]]
            c = self.node_positions_dict_2d[line_2[0]]
            d = self.node_positions_dict_2d[line_2[1]]

            min_dist, crossing_position = compute_line_segment_intersection(a, b, c, d)

            if min_dist > 0.0001:
                crossing_position = None

            if crossing_position is np.inf:
                raise ValueError('The edges are overlapping. This is not a valid spatial graph.')

            elif crossing_position is None:
                pass

            elif type(crossing_position) is np.ndarray and min_dist < 0.0001:
                crossings.append('crossing_' + str(crossing_num))
                crossing_num += 1
                crossing_positions.append(crossing_position)
                crossing_edge_pair = self.get_crossing_edge_order(line_1, line_2, crossing_position)
                crossing_edge_pairs.append(crossing_edge_pair)

        return crossings, crossing_positions, crossing_edge_pairs

    def get_crossings_3D(self):
        """
        Get the 3D position of a crossing.
        """

        crossings, crossing_positions_2D, crossing_edge_pairs = self.get_crossings()

        # Convert the 2D crossing positions to 3D crossing positions
        xz_coords = np.array(crossing_positions_2D)
        y_coords = np.zeros((xz_coords.shape[0], 1))
        crossing_positions_2D = np.hstack((xz_coords, y_coords))
        crossing_positions_2D = crossing_positions_2D[:, [0, 2, 1]]

        nodes_dict = {node: self.node_positions_3d[self.nodes.index(node)] for node in self.nodes}

        crossing_positions_3D = []
        for crossing, crossing_position_2D,crossing_edge_pair in zip(crossings, crossing_positions_2D, crossing_edge_pairs):
            edge_1, edge_2 = crossing_edge_pair
            node_0, node_2 = edge_1
            node_1, node_3 = edge_2
            node_0_position = nodes_dict[node_0]
            node_1_position = nodes_dict[node_1]
            node_2_position = nodes_dict[node_2]
            node_3_position = nodes_dict[node_3]
            crossing_position_3D_02, crossing_position_3D_13 = compute_3D_intersection(node_0_position, node_1_position, node_2_position, node_3_position, crossing_position_2D)
            crossing_positions_3D.append((crossing_position_3D_02, crossing_position_3D_13))


        crossing_positions_2D_dict = {crossing: position for crossing, position in zip(crossings, crossing_positions_2D)}

        crossing_positions_3D_dict = {}
        for crossing, crossing_position_2D, crossing_position_3D, crossing_edge_pair in zip(crossings,
                                                                                            crossing_positions_2D,
                                                                                            crossing_positions_3D,
                                                                                            crossing_edge_pairs):
            edge_pair_1, edge_pair_2 = crossing_edge_pair
            node_a, node_c = edge_pair_1
            node_b, node_d = edge_pair_2

            crossing_position_3D_ac, crossing_position_3D_bd = crossing_position_3D
            crossing_positions_3D_dict[crossing] = {'projection': crossing_position_2D,
                                                    node_a: crossing_position_3D_ac,
                                                    node_b: crossing_position_3D_bd,
                                                    node_c: crossing_position_3D_ac,
                                                    node_d: crossing_position_3D_bd}


        return crossings, crossing_positions_2D, crossing_positions_3D, crossing_edge_pairs, crossing_positions_3D_dict




    def project(self, node_positions, max_iter=2, forced_rotation=None):
        """
        Project the spatial graph onto a random 2D plane.

        The projection is done by applying a random 3D rotation to the graph and then projecting it onto a 2D plane.
        For simplicity, we use the transformed XZ plane. When the rotation is complete, the program checks that the
        projected graph for several things.

        First, it checks to make sure that no combinations of edges and/or nodes are overlapping.

        Second, it checks to make sure that no edges are perfectly vertical or perfectly horizontal. While neither of
        these cases technically incorrect, it's easier to implement looping through rotations rather than add edge cases
        in for these cases.

        """

        # Define the random rotations (in a deterministic manner w/ a Halton sequence)
        sampler = qmc.Halton(d=3, scramble=False)
        halton_samples = sampler.random(n=max_iter)
        halton_rotations = 2 * np.pi * halton_samples
        rotations = halton_rotations

        # Add the initial rotation if it exists, for debugging purposes
        if forced_rotation is not None:
            rotations = np.vstack((forced_rotation, rotations))

        for rotation in rotations:
            try:
                # Rotate the node positions
                rotated_node_positions = rotate(node_positions, rotation)
                projected_node_positions = rotated_node_positions[:, [0, 2]]

                # First, check that no edges are perfectly vertical or perfectly horizontal.
                # While neither of these cases technically incorrect, it's easier to implement looping through rotations
                # rather than add edge cases for each 2D and 3D line equation.

                for edge in self.edges:
                    x1, z1 = projected_node_positions[self.nodes.index(edge[0])]
                    x2, z2 = projected_node_positions[self.nodes.index(edge[1])]

                    if x1 == x2 or z1 == z2:
                        raise ValueError('An edge is vertical or horizontal. This is not a valid spatial graph.')

                # Second, check adjacent edge pairs for validity.
                # Since adjacent segments are straight lines, they should only intersect at a single endpoint.
                # The only other possibility is for them to infinitely overlap, which is not a valid spatial graph.

                for line_1, line_2 in self.adjacent_edge_pairs:
                    a = projected_node_positions[self.nodes.index(line_1[0])]
                    b = projected_node_positions[self.nodes.index(line_1[1])]
                    c = projected_node_positions[self.nodes.index(line_2[0])]
                    d = projected_node_positions[self.nodes.index(line_2[1])]
                    min_dist, crossing_position, = compute_line_segment_intersection(a, b, c, d)

                    if min_dist > 0.0001:
                        crossing_position = None

                    if crossing_position is not None and crossing_position is not np.inf:
                        assertion_1 = all((crossing_position - a) < 0.0001)
                        assertion_2 = all((crossing_position - b) < 0.0001)
                        assertion_3 = all((crossing_position - c) < 0.0001)
                        assertion_4 = all((crossing_position - d) < 0.0001)
                        if not any([assertion_1, assertion_2, assertion_3, assertion_4]):
                            raise ValueError('Adjacent edges must intersect at the endpoints.')

                    elif crossing_position is np.inf:
                        raise ValueError('The edges are overlapping. This is not a valid spatial graph.')

                # If all are satisfied
                break

            except ValueError:
                continue

        rotated_node_positions = rotate(node_positions, rotation)
        return rotated_node_positions

    def create_spatial_graph_diagram(self):

        nodes_and_crossings = self.nodes + self.crossings

        # Create the vertex and crossing objects
        # len([edge for edge in self.edges if node in edge])
        vertex_node_degrees = [len([edge for edge in self.edges if node in edge]) for node in self.nodes]
        vertices = [Vertex(degree, 'node_' + node) for node, degree in zip(self.nodes,vertex_node_degrees)]
        if self.crossing_positions is not None:
            crossings = [Crossing('crossing_object_' + str(i)) for i in range(len(self.crossing_positions))]
        else:
            crossings = []
        vertices_and_crossings = vertices + crossings

        # Create a dictionary that contains the cyclical ordering of every node and crossing
        node_ordering_dict = self.cyclic_order_vertices()
        crossing_ordering_dict = self.cyclic_order_crossings()

        cyclic_ordering_dict = {**node_ordering_dict, **crossing_ordering_dict}

        for sub_edge in self.get_sub_edges():
            node_a, node_b = sub_edge

            node_a_index = nodes_and_crossings.index(node_a)
            node_b_index = nodes_and_crossings.index(node_b)

            vertex_a = vertices_and_crossings[node_a_index]
            vertex_b = vertices_and_crossings[node_b_index]

            if not vertex_a.already_assigned(vertex_b) and not vertex_b.already_assigned(vertex_a):

                vertex_b_index_for_vertex_a = cyclic_ordering_dict[node_a][node_b]
                vertex_a_index_for_vertex_b = cyclic_ordering_dict[node_b][node_a]

                vertex_a[vertex_b_index_for_vertex_a] = vertex_b[vertex_a_index_for_vertex_b]

            else:
                raise ValueError('The vertices are already assigned.')


        sgd = SpatialGraphDiagram(vertices=vertices, crossings=crossings)

        return sgd

    def plot(self):

        edges = self.edges
        nodes = self.nodes
        node_positions = self.node_positions_3d
        crossings = self.crossings
        crossing_positions = self.crossing_positions
        node_positions_dict = {node: position for node, position in zip(nodes, node_positions)}
        contiguous_sub_edges, contiguous_sub_edge_positions = self.get_contiguous_edges()

        p = plot_spatial_graph(nodes, node_positions, edges,contiguous_sub_edges, contiguous_sub_edge_positions)

        p.show()

