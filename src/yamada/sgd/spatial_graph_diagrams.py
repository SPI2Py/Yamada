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
import matplotlib
matplotlib.use('TkAgg')   # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from yamada.poly.H_polynomial import h_poly
from yamada.poly.utilities import get_coefficients_and_exponents, normalize_poly
from yamada.sgd.diagram_elements import Vertex, Edge, Crossing


class SpatialGraphDiagram:
    """
    TODO: Fix labels. Normalize them, and make sure they are unique (both currently, and going forward)
    """

    def __init__(self, *, edges=None, vertices=None, crossings=None, check=True):

        # Ensure inputs are lists (avoid mutable default arguments)
        edges = edges or []
        vertices = vertices or []
        crossings = crossings or []

        # Combine all elements into a single list
        data = edges + vertices + crossings

        # Ensure labels are unique by creating a dictionary
        self.data = {d.label: d for d in data}

        # Validate label uniqueness
        if len(data) != len(self.data):
            raise ValueError("Labels must be unique across all diagram elements.")

        # Categorize elements by type
        self.edges = [d for d in data if isinstance(d, Edge)]
        self.vertices = [d for d in data if isinstance(d, Vertex)]
        self.crossings = [d for d in data if isinstance(d, Crossing)]

        # Optional: Validate categorization
        if len(self.edges) != len(edges):
            raise ValueError("Some edges are incorrectly classified.")
        if len(self.vertices) != len(vertices):
            raise ValueError("Some vertices are incorrectly classified.")
        if len(self.crossings) != len(crossings):
            raise ValueError("Some crossings are incorrectly classified.")

        # Normalize labels
        self._normalize_labels()

        # Ensure that vertices and crossings are connected indirectly via edges
        self._preprocess_diagram()

        # Add edges and 2-valent vertices where necessary to ensure that self-loops have a planar embedding
        # self._inflate_self_loops()

        # Optionally run additional checks
        if check:
            self._check()

    def _normalize_labels(self):
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

    def _preprocess_diagram(self):
        """
        Ensures that the diagram is correctly assembled.
        """

        # Pairs of edges must be connected by a vertex.
        for A in self.edges:
            for i in range(2):
                B, j = A.adjacent[i]
                if isinstance(B, Edge):
                    self._create_vertex((A, i), (B, j))

        # Pairs of vertices and/or crossings must be connected by an edge.
        for A in self.crossings + self.vertices:
            for i in range(A.degree):
                B, j = A.adjacent[i]
                if not isinstance(B, Edge):
                    self._create_edge(A, i, B, j)

    def _check(self):
        """
        Checks that the diagram is valid.
        """

        assert 2 * len(self.edges) == sum(d.degree for d in self.crossings + self.vertices)

        for C in self.crossings:
            assert all(isinstance(v, Edge) for v, j in C.adjacent)
        for V in self.vertices:
            assert all(isinstance(v, Edge) for v, j in V.adjacent)
        for E in self.edges:
            assert all(not isinstance(v, Edge) for v, j in E.adjacent)

        # Graph is planar
        assert self._is_planar()

    def _euler(self):
        """
        Returns the Euler characteristic of the diagram.
        """
        v = len(self.crossings) + len(self.vertices)
        e = len(self.edges)
        f = len(self.faces())
        return v - e + f

    def _is_planar(self):
        """
        Returns True if the diagram is planar.
        """
        return self._euler() == 2 * len(list(nx.connected_components(self.projection_graph())))



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

        # Ensure the edges share a 2-valent vertex
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
        (A, i), (B, j) = keep_edge.adjacent
        (C, k), (D, l) = remove_edge.adjacent
        if A == C:
            keep_edge_connect_idx = 0
            remove_vertex = C
            remove_edge_keep_obj = D
            remove_edge_keep_idx = l
        elif A == D:
            keep_edge_connect_idx = 0
            remove_vertex = D
            remove_edge_keep_obj = C
            remove_edge_keep_idx = k
        elif B == C:
            keep_edge_connect_idx = 1
            remove_vertex = C
            remove_edge_keep_obj = D
            remove_edge_keep_idx = l
        elif B == D:
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

    def _simplify(self):
        """
        Simplifies the diagram by removing unnecessary edges and vertices.
        """

        # Remove unnecessary edges
        for V in self.vertices:
            if V.degree == 2:
                (A, i), (B, j) = V.adjacent
                if isinstance(A, Edge) and isinstance(B, Edge):
                    self._merge_edges(A, B)

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

    def projection_graph(self):
        """
        TODO Add documentation
        """

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
            resolved_diagram._check()

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

    # def calculate_yamada_polynomial(self, check_pieces=False):
    #     """
    #     Recursively calculates the Yamada polynomial of the spatial graph diagram.
    #
    #     Args:
    #         check_pieces (bool): If True, validates intermediate diagrams.
    #
    #     Returns:
    #         pari: The calculated Yamada polynomial.
    #     """
    #     A = pari('A')
    #
    #     # Base case: no crossings left
    #     if len(self.crossings) == 0:
    #         return h_poly(self.projection_graph())
    #
    #     # Recursive case: handle the first crossing
    #     crossing = self.crossings[0]
    #     S_plus = self._resolve_crossing(crossing, "S_plus", check_pieces)
    #     S_minus = self._resolve_crossing(crossing, "S_minus", check_pieces)
    #     S_0 = self._resolve_crossing(crossing, "S_0", check_pieces)
    #
    #     # Combine the polynomials using the Yamada polynomial formula
    #     Y_plus = S_plus.calculate_yamada_polynomial()
    #     Y_minus = S_minus.calculate_yamada_polynomial()
    #     Y_0 = S_0.calculate_yamada_polynomial()
    #
    #     return A * Y_plus + (A ** -1) * Y_minus + Y_0
    #
    #
    # def yamada_polynomial(self, normalize=True):
    #     """normalized_yamada_polynomial"""
    #
    #     yamada_polynomial = self.calculate_yamada_polynomial()
    #
    #     if normalize:
    #         yamada_polynomial =  normalize_poly(yamada_polynomial)
    #
    #     return yamada_polynomial

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
            yamada_poly = h_poly(self.projection_graph())
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
        Creates a planar embedding of the spatial graph diagram by introducing intermediate nodes
        and labeling intermediate edges and nodes.

        Returns:
            G: Planar-friendly NetworkX graph.
            node_labels: Dictionary of labels for intermediate nodes.
            edge_labels: Dictionary of labels for intermediate edges.
        """
        G = nx.Graph()
        node_labels = {}
        edge_labels = {}

        # Add nodes for vertices, crossings, and edges with their type
        for crossing in self.crossings:
            G.add_node(crossing.label, type="Crossing")
        for vertex in self.vertices:
            G.add_node(vertex.label, type="Vertex")
        for edge in self.edges:
            G.add_node(edge.label, type="Edge")

        # Add intermediate nodes and label intermediate edges
        intermediate_counter = 0
        for edge in self.edges:
            for i, (connected_obj, index) in enumerate(edge.adjacent):
                # Create an intermediate node
                intermediate_node = f"int_{intermediate_counter}"
                intermediate_counter += 1

                # Add intermediate node with its type
                G.add_node(intermediate_node, type="Intermediate")

                # Create labeled edges
                intermediate_edge_1 = f"{edge.label}[{i}]"
                intermediate_edge_2 = f"{connected_obj.label}[{index}]"

                # Connect intermediate edges with intermediate node
                G.add_edge(edge.label, intermediate_node, label=i)  # Edge index
                G.add_edge(intermediate_node, connected_obj.label, label=index)  # Edge index

                # Save labels for intermediate edges
                edge_labels[(edge.label, intermediate_node)] = str(i)
                edge_labels[(intermediate_node, connected_obj.label)] = str(index)

                # Save labels for intermediate nodes
                node_labels[intermediate_node] = f"{edge.label}[{i}]={connected_obj.label}[{index}]"

        return G, node_labels, edge_labels

    def plot(self, highlight_nodes=None, highlight_labels=None):
        """
        Plots the spatial graph diagram, labeling intermediate edges with index numbers
        and intermediate nodes with full index assignments.
        """

        # Step 1: Create the planar-friendly graph
        planar_graph, node_labels, edge_labels = self.planar_embedding()

        # Step 2: Generate the planar embedding
        is_planar, embedding = nx.check_planarity(planar_graph)
        if not is_planar:
            raise ValueError("The graph is not planar!")
        pos = nx.planar_layout(embedding)

        # Step 3: Separate node types
        edges = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Edge"]
        vertices = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Vertex"]
        crossings = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Crossing"]
        regular_nodes = [n for n, d in planar_graph.nodes(data=True) if d["type"] != "Intermediate"]
        intermediate_nodes = [n for n, d in planar_graph.nodes(data=True) if d["type"] == "Intermediate"]

        # Initialize the plot
        plt.figure(figsize=(12, 12))

        # Draw the edges
        nx.draw_networkx_edges(planar_graph, pos)

        # Label the edges
        nx.draw_networkx_edge_labels(
            planar_graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="black",
            rotate=False,
        )

        # Draw the nodes
        nx.draw_networkx_nodes(
            planar_graph,
            pos,
            nodelist=edges,
            node_color="gray",
            node_size=300,
            alpha=0.7
        )

        nx.draw_networkx_nodes(
            planar_graph,
            pos,
            nodelist=vertices,
            node_color="lightgreen",
            node_size=300,
            alpha=0.7
        )

        nx.draw_networkx_nodes(
            planar_graph,
            pos,
            nodelist=crossings,
            node_color="lightblue",
            node_size=300,
            alpha=0.7
        )

        # Label the nodes
        nx.draw_networkx_labels(
            planar_graph,
            pos,
            labels={n: n for n in regular_nodes},
            font_size=10,
        )

        if highlight_nodes:
            nx.draw_networkx_nodes(
                planar_graph,
                pos,
                nodelist=highlight_nodes,
                node_color="yellow",
                node_size=2000,
                label="Highlighted Nodes",
                alpha=0.4,
                edgecolors="orange"
            )
            # nx.draw_networkx_labels(
            #     planar_graph,
            #     pos,
            #     labels={n: highlight_labels.get(n, n) for n in highlight_nodes},
            #     font_size=12,
            #     font_color="white",
            #     font_weight="bold",
            # )

        # State all object-index assignments
        intermediate_label_text = "Object-Index Pairs \n" + "\n".join(f"{label}" for node, label in node_labels.items())
        plt.gcf().text(
            0.85, 0.5,  # Position the text box to the right of the plot
            intermediate_label_text,
            fontsize=10,
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.9),
        )

        # Step 10: Remove axis and spines
        plt.axis("off")  # Turn off axes, ticks, and labels
        ax = plt.gca()  # Get the current axis
        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide all spines

        # Show the plot
        plt.title("Planar Embedding of the Spatial Graph Diagram")
        plt.show()
