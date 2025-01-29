"""
Enumerating spatial graph diagrams
==================================

Current code is limited to trivalent system architectures.

The basic approach differs somewhat from the one in the paper.
Namely, I use "plantri" to enumerate possible diagram shadows with the
specified number of crossings::

http://users.cecs.anu.edu.au/~bdm/plantri/

You need to compile plantri and have it in the same directory as this
file (or somewhere in your path) for the enumeration to work.

Due to a limitation of plantri, this restricts us to shadows which are
"diagrammatically prime" in that there is not a circle meeting the
shadow in two points that has vertices of the shadow on both sides.
Equivalently, the dual planar graph is simple.

If the system architecture graph cannot be disconnected by removing
two edges, this only excludes shadows all of whose spatial diagrams
are the connect sum of a spatial graph diagram with the desired system
architecture and a knot.  Presumably, we would want to exclude such in
any event.  However, the example in Case Study 1 can be so
disconnected...

Validation
==========

Compared to Dobrynin and Vesnin:

1. For the theta graph, the list of Yamada polynomials through 5
   crossings matches after removing the non-prime examples from their
   list (theta_3, theta_5, theta_10, theta_14).

2. For the tetrahedral graph, the list of Yamada polynomials through 4
   crossings matches after removing the non-prime Omega_5.

Note: The way this script is written w/ pickling you must import this script into another script
rather than directly calculate Yamada polynomials in this script (you'll get error messages)
"""

import networkx as nx
import pickle
from cypari import pari
import warnings
import matplotlib
matplotlib.use('TkAgg')   # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
import numpy as np

from yamada.poly.H_polynomial import h_poly
from yamada.poly.utilities import get_coefficients_and_exponents, normalize_poly
from yamada.sgd.diagram_elements import Vertex, Edge, Crossing
from ..utils.visualization import plot_spatial_graph_diagram

class SpatialGraphDiagram:

    def __init__(self, *, edges=None, vertices=None, crossings=None,
                 standardize_labels=True,
                 correct_diagram=True,
                 simplify_diagram=True):

        # Ensure inputs are lists (avoid mutable default arguments)
        edges = edges or []
        vertices = vertices or []
        crossings = crossings or []

        # Validate inputs
        self._validate_inputs(edges, vertices, crossings)

        # Categorize elements by type
        self.edges = edges
        self.vertices = vertices
        self.crossings = crossings
        self.data = {d.label: d for d in self.edges + self.vertices + self.crossings}

        if standardize_labels:
            self._standardize_labels()

        if correct_diagram:
            self._correct_diagram()

        if simplify_diagram:
            self.simplify_diagram()

    @staticmethod
    def _validate_inputs(edges, vertices, crossings):

        data = edges + vertices + crossings

        # Ensure all elements are edges
        e_are_e = all(isinstance(e, Edge) for e in edges)
        v_are_v = all(isinstance(v, Vertex) for v in vertices)
        c_are_c = all(isinstance(c, Crossing) for c in crossings)
        if not e_are_e or not v_are_v or not c_are_c:
            raise TypeError("Some elements are incorrectly categorized.")

        # Ensure all elements are unique
        e_are_unique = len(edges) == len(set(e.label for e in edges))
        v_are_unique = len(vertices) == len(set(v.label for v in vertices))
        c_are_unique = len(crossings) == len(set(c.label for c in crossings))
        if not e_are_unique or not v_are_unique or not c_are_unique:
            raise ValueError("Labels must be unique.")

        # Ensure all indices are assigned to other diagram elements
        all_indices = []
        unique_indices = set()
        for element in data:
            for i, adjacent in enumerate(element.adjacent):
                if adjacent is None:
                    raise ValueError(f"Object {element.label} index {i} is not assigned.")
                elif adjacent[0] not in data:
                    raise ValueError(f"Object {element.label} index {i} is incorrectly assigned to an object not within the SGD.")
                else:
                    all_indices.append(adjacent)
                    unique_indices.add(adjacent)


        # Ensure all indices are uniquely assigned.
        if len(all_indices) != len(unique_indices):
            raise ValueError("Indices must be unique.")

        return edges, vertices, crossings

    def _correct_diagram(self):
        """
        Ensures that the diagram is correctly assembled.
        """

        if len(self.edges) == 0 and len(self.vertices) == 0 and len(self.crossings) == 0:
            warnings.warn("Empty diagram.", UserWarning)

        # Pairs of edges must be connected by a vertex.
        for A in self.edges:
            for i in range(2):
                B, j = A.adjacent[i]
                if isinstance(B, Edge):
                    warnings.warn(f"Edges {A.label}[{i}] and {B.label}[{j}] should be connected by a two-valent vertex.", UserWarning)
                    self._create_vertex((A, i), (B, j))

        # Pairs of vertices and/or crossings must be connected by an edge.
        for A in self.crossings + self.vertices:
            for i in range(A.degree):
                B, j = A.adjacent[i]
                if not isinstance(B, Edge):
                    warnings.warn(
                        f"Vertices {A.label}[{i}] and {B.label}[{j}] should be connected by an edge.",
                        UserWarning)
                    self._create_edge(A, i, B, j)

        # Check the corrected diagram
        self.check()

    def _standardize_labels(self):
        """
        Renumbers labels for edges, vertices, and crossings sequentially.
        """
        # Renumber edges
        for i, edge in enumerate(self.edges, start=1):
            old_label = edge.label
            new_label = f"e{i}"
            edge.label = new_label
            self.data.pop(old_label)
            self.data[new_label] = edge

        # Renumber vertices
        for i, vertex in enumerate(self.vertices, start=1):
            old_label = vertex.label
            new_label = f"v{i}"
            vertex.label = new_label
            self.data.pop(old_label)
            self.data[new_label] = vertex

        # Renumber crossings
        for i, crossing in enumerate(self.crossings, start=1):
            old_label = crossing.label
            new_label = f"c{i}"
            crossing.label = new_label
            self.data.pop(old_label)
            self.data[new_label] = crossing

        # Update counters
        self.edge_counter = len(self.edges)
        self.vertex_counter = len(self.vertices)
        self.crossing_counter = len(self.crossings)

    def simplify_diagram(self):
        """Merges edges sharing 2-valent vertices until no more simplifications can be made."""

        changed = True
        while changed:
            changed = False

            # Iterate over vertices in reverse order
            for V in reversed(self.vertices):  # Process higher-labeled vertices first
                if V.degree == 2:
                    (A, i), (B, j) = V.adjacent  # Adjacent edges

                    # Ensure both are edges
                    if not (isinstance(A, Edge) and isinstance(B, Edge)):
                        raise ValueError("Vertices must be connected to edges.")

                    # Merge the two edges
                    if A != B:  # Only merge if the edges are different
                        self._merge_edges(A, B)
                        changed = True
                        break  # Restart the loop after a change

        # Check the simplified diagram
        self.check()

    def check(self):

        # Check that the diagram is connected
        assert 2 * len(self.edges) == sum(d.degree for d in self.crossings + self.vertices)

        # Check that the graph is planar
        v = len(self.crossings) + len(self.vertices)
        e = len(self.edges)
        f = len(self.faces())
        euler = v - e + f
        is_planar = euler == 2 * len(list(nx.connected_components(self.graph())))
        assert is_planar


    def __eq__(self, other):

        # Check the individual elements

        if not isinstance(other, SpatialGraphDiagram):
            raise ValueError("This must be a SpatialGraphDiagram.")

        # Compare edges
        edges_s = [edge.label for edge in self.edges]
        edges_o = [edge.label for edge in other.edges]

        if len(edges_s) != len(edges_o):
            return False

        for edge_s, edge_o in zip(sorted(edges_s), sorted(edges_o)):
            if edge_s != edge_o:
                return False

        # Compare vertices
        vertices_s = [vertex.label for vertex in self.vertices]
        vertices_o = [vertex.label for vertex in other.vertices]

        if len(vertices_s) != len(vertices_o):
            return False

        for vertex_s, vertex_o in zip(sorted(vertices_s), sorted(vertices_o)):
            if vertex_s != vertex_o:
                return False

        # Compare crossings
        crossings_s = [crossing.label for crossing in self.crossings]
        crossings_o = [crossing.label for crossing in other.crossings]

        if len(crossings_s) != len(crossings_o):
            return False

        for crossing_s, crossing_o in zip(sorted(crossings_s), sorted(crossings_o)):
            if crossing_s != crossing_o:
                return False

        # Check the index assignments

        for obj_s, obj_o in zip(self.data.values(), other.data.values()):
            for i in range(obj_s.degree):
                adj_s, idx_s = obj_s.adjacent[i]
                adj_o, idx_o = obj_o.adjacent[i]
                if adj_s.label != adj_o.label or idx_s != idx_o:
                    return False

        return True

    def __neq__(self, other):
        return not self == other

    def _create_edge(self, A, i, B, j):
        """Creates and adds an edge to the diagram."""

        # Create a new edge
        self.edge_counter += 1
        edge_label = f'e{self.edge_counter}'
        edge = Edge(edge_label)

        # Add the edge to the diagram
        self._add_edge(edge)

        # Connect the edge to the diagram elements
        self.connect(A, i, edge, 0)
        self.connect(B, j, edge, 1)

    def _create_vertex(self, *args):
        """Creates and adds an n-valent vertex to the diagram. Each argument is a tuple (A,i), (B, j), etc."""

        # Create a new vertex
        self.vertex_counter += 1
        vertex_label = f'v{self.vertex_counter}'
        vertex = Vertex(len(args), vertex_label)

        # Add the vertex to the diagram
        self._add_vertex(vertex)

        # Connect the vertex to the diagram elements
        for i, (obj, idx) in enumerate(args):
            self.connect(obj, idx, vertex, i)

    def _create_crossing(self, A, i, B, j, C, k, D, l):
        """Creates and adds a crossing to the diagram."""

        # Create a new crossing
        self.crossing_counter += 1
        crossing_label = f'c{self.crossing_counter}'
        crossing = Crossing(crossing_label)

        # Add the crossing to the diagram
        self._add_crossing(crossing)

        # Connect the crossing to the diagram elements
        self.connect(A, i, crossing, 0)
        self.connect(B, j, crossing, 1)
        self.connect(C, k, crossing, 2)
        self.connect(D, l, crossing, 3)

    def _add_edge(self, edge):
        """Adds an edge to the diagram."""
        self.edges.append(edge)
        self.data[edge.label] = edge

    def _add_vertex(self, vertex):
        """Adds a vertex to the diagram."""
        self.vertices.append(vertex)
        self.data[vertex.label] = vertex

    def _add_crossing(self, crossing):
        """Adds a crossing to the diagram."""
        self.crossings.append(crossing)
        self.data[crossing.label] = crossing

    def _remove_edge(self, edge):
        """Removes an edge from the diagram."""
        self.edges.remove(edge)
        self.data.pop(edge.label)

    def _remove_vertex(self, vertex):
        """Removes a vertex from the diagram."""
        self.vertices.remove(vertex)
        self.data.pop(vertex.label)

    def _remove_crossing(self, crossing):
        """Removes a crossing from the diagram."""
        self.crossings.remove(crossing)
        self.data.pop(crossing.label)

    def _merge_edges(self, E0, E1):
        """
        Merges two edges with a shared 2-valent vertex into a single edge. Keeps the label with the lowest value.
        """

        # Ensure that the diagram elements are edges
        assert isinstance(E0, Edge) and isinstance(E1, Edge)
        assert E0 != E1

        # Ensure the edges share at least one 2-valent vertex
        assert any(adj0 == adj1 for adj0, _ in E0.adjacent for adj1, _ in E1.adjacent)

        # Convert the edge labels into integers
        E0_label = int(E0.label[1:])
        E1_label = int(E1.label[1:])
        if E0_label < E1_label:
            keep_edge = E0
            remove_edge = E1
        elif E0_label > E1_label:
            keep_edge = E1
            remove_edge = E0
        else:
            raise ValueError('Edges must have distinct labels.')

        # Find the shared 2-valent vertex
        # If the edges share a two 2-valent then select the higher index
        (A, i), (B, j) = keep_edge.adjacent
        (C, k), (D, l) = remove_edge.adjacent

        # If the edges shared two vertices, remove the higher vertex
        edges_share_two_objects = (A == C and B == D) or (A == D and B == C)
        objects_are_vertices = isinstance(A, Vertex) and isinstance(B, Vertex)
        shared_vertices_are_two_valent = A.degree == 2 and B.degree == 2
        if edges_share_two_objects and objects_are_vertices and shared_vertices_are_two_valent:
            A_label = int(A.label[1:])
            B_label = int(B.label[1:])
            if A_label > B_label:
                remove_vertex = A
                keep_edge_connect_idx = 1 if A == B else 0  # Connect to the other vertex
            else:
                remove_vertex = B
                keep_edge_connect_idx = 0 if A == B else 1  # Connect to the other vertex
            remove_edge_keep_obj = D if remove_vertex == C else C
            remove_edge_keep_idx = l if remove_vertex == C else k
        elif A == C and isinstance(A, Vertex):
            keep_edge_connect_idx = 0
            remove_vertex = C
            remove_edge_keep_obj = D
            remove_edge_keep_idx = l
        elif A == D and isinstance(A, Vertex):
            keep_edge_connect_idx = 0
            remove_vertex = D
            remove_edge_keep_obj = C
            remove_edge_keep_idx = k
        elif B == C and isinstance(B, Vertex):
            keep_edge_connect_idx = 1
            remove_vertex = C
            remove_edge_keep_obj = D
            remove_edge_keep_idx = l
        elif B == D and isinstance(B, Vertex):
            keep_edge_connect_idx = 1
            remove_vertex = D
            remove_edge_keep_obj = C
            remove_edge_keep_idx = k
        else:
            raise ValueError("Edges must share a 2-valent vertex.")

        # Update the diagram
        self._remove_edge(remove_edge)
        self._remove_vertex(remove_vertex)
        self.connect(keep_edge, keep_edge_connect_idx, remove_edge_keep_obj, remove_edge_keep_idx)

    def faces(self):
        """
        The faces are the complementary regions of the diagram. Each
        face is given as a list of corners of BaseVertices as one goes
        around *clockwise*. These corners are recorded as
        EntryPoints, where EntryPoints(c, j) denotes the corner of the
        face abutting crossing c between strand j and j + 1.

        Alternatively, the sequence of EntryPoints can be regarded as
        the *heads* of the oriented edges of the face.
        """

        entry_points = []

        for V in self.data.values():
            entry_points += V.entry_points()

        corners = set(entry_points)
        faces = []

        while len(corners):
            face = [corners.pop()]
            while True:
                next_ep = face[-1].next_corner()
                if next_ep == face[0]:
                    faces.append(face)
                    break
                else:
                    corners.remove(next_ep)
                    face.append(next_ep)

        return faces

    def get_object(self, label):
        return self.data[label]

    def connect(self, A, i, B, j):

        A_is_edge = isinstance(A, Edge)
        B_is_edge = isinstance(B, Edge)

        # If both diagram elements are edges, then connect them with a vertex.
        if A_is_edge and B_is_edge:
            self._create_vertex((A, i), (B, j))

        # If only one diagram element is an edge, then connect them directly.
        elif (A_is_edge and not B_is_edge) or (not A_is_edge and B_is_edge):
            A[i] = B[j]

        # If neither diagram element is an edge, then connect them with an edge.
        else:
            self._create_edge(A, i, B, j)


    def short_cut(self, crossing, i0):
        """
        Short-cuts a crossing by removing the edge between them.
        """

        i1 = (i0 + 1) % 4
        E0, j0 = crossing.adjacent[i0]
        E1, j1 = crossing.adjacent[i1]
        if E0 == E1:
            self._create_vertex((E0, j0), (E1, j1))
        else:
            self.connect(E0, j0, E1, j1)


    def copy(self):
        """
        Returns a serialized copy of the diagram.
        """
        return pickle.loads(pickle.dumps(self))

    def graph(self):
        G = nx.MultiGraph()
        for e in self.edges:
            v = e.adjacent[0][0].label
            w = e.adjacent[1][0].label
            G.add_edge(v, w)
        return G

    def underlying_graph(self):
        """
        TODO Add documentation
        """

        G = nx.MultiGraph()
        vertex_inputs = set()

        for V in self.vertices:
            vertex_inputs.update((V, i) for i in range(V.degree))

        edges_used = 0

        while len(vertex_inputs):
            V, i = vertex_inputs.pop()
            W, j = V.adjacent[i]
            while not isinstance(W, Vertex):
                if isinstance(W, Edge):
                    edges_used += 1
                W, j = W.flow(j)
            vertex_inputs.remove((W, j))
            v, w = V.label, W.label
            G.add_edge(v, w)

        if edges_used == len(self.edges):
            return G

    def _resolve_crossing(self, crossing, resolution_type, check_pieces=False):
        """
        Resolves a crossing into a simplified diagram based on the resolution type.

        Args:
            crossing (Crossing): The crossing to resolve.
            resolution_type (str): One of "S_plus", "S_minus", or "S_0".
            check_pieces (bool): If True, validates the resolved diagram.

        Returns:
            SpatialGraphDiagram: The resolved spatial graph diagram.
        """
        resolved_diagram = self.copy()
        crossing_copy = resolved_diagram.data[crossing.label]
        resolved_diagram._remove_crossing(crossing_copy)

        if resolution_type == "S_plus":
            resolved_diagram.short_cut(crossing_copy, 0)
            resolved_diagram.short_cut(crossing_copy, 2)
        elif resolution_type == "S_minus":
            resolved_diagram.short_cut(crossing_copy, 1)
            resolved_diagram.short_cut(crossing_copy, 3)
        elif resolution_type == "S_0":
            resolved_diagram._create_s0_vertex(crossing_copy)
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

        if check_pieces:
            resolved_diagram.check()

        return resolved_diagram

    def _create_s0_vertex(self, crossing):
        """
        Creates a 4-valent vertex to replace a crossing in the S_0 resolution.

        Args:
            crossing (Crossing): The crossing being replaced.
        """
        # Create a new vertex
        vertex_label = f"{crossing.label}_smushed"
        new_vertex = Vertex(4, vertex_label)
        self._add_vertex(new_vertex)

        # Connect the adjacent elements to the new vertex
        for i in range(4):
            connected_obj, index = crossing.adjacent[i]
            new_vertex[i] = connected_obj[index]

    def yamada_polynomial(self, normalize=True, check_pieces=False):
        """
        Calculates the Yamada polynomial of the spatial graph diagram, optionally normalizing it.

        Args:
            normalize (bool): If True, normalize the Yamada polynomial.
            check_pieces (bool): If True, validates intermediate diagrams during calculation.

        Returns:
            pari: The calculated (and optionally normalized) Yamada polynomial.
        """
        A = pari('A')

        # Base case: no crossings left
        if len(self.crossings) == 0:
            yamada_poly = h_poly(self.graph())
        else:
            # Recursive case: handle the first crossing
            crossing = self.crossings[0]
            S_plus = self._resolve_crossing(crossing, "S_plus", check_pieces)
            S_minus = self._resolve_crossing(crossing, "S_minus", check_pieces)
            S_0 = self._resolve_crossing(crossing, "S_0", check_pieces)

            # Combine the polynomials using the Yamada polynomial formula
            Y_plus = S_plus.yamada_polynomial(normalize=False, check_pieces=check_pieces)
            Y_minus = S_minus.yamada_polynomial(normalize=False, check_pieces=check_pieces)
            Y_0 = S_0.yamada_polynomial(normalize=False, check_pieces=check_pieces)

            yamada_poly = A * Y_plus + (A ** -1) * Y_minus + Y_0

        # Normalize if required
        if normalize:
            yamada_poly = normalize_poly(yamada_poly)

        return yamada_poly

    def planar_embedding(self):
        """
        Returns the underlying planar embedding of an abstract graph.
        """

        G = nx.PlanarEmbedding()
        parts = self.vertices + self.crossings + self.edges

        for A in parts:
            for i in range(A.degree):
                B, j = A.adjacent[i]
                ref_nbr = None if i == 0 else A.adjacent[i - 1][0].label
                G.add_half_edge_ccw(A.label, B.label, ref_nbr)

        G.check_structure()
        return G

    def plot(self):

        p = plot_spatial_graph_diagram(self)

        # planar_graph = self.planar_embedding()
        # p = plot_spatial_graph_diagram(planar_graph)
        p.show()

