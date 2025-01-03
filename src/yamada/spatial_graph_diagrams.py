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
from yamada.H_polynomial import h_poly
from yamada.utilities import get_coefficients_and_exponents
from yamada.diagram_elements import Vertex, Edge, Crossing


class SpatialGraphDiagram:
    """

    """

    def __init__(self, data, check=True):

        self.edges = None
        self.crossings = None
        self.vertices = None

        # Need labels of vertices/crossings to be unique and hashable
        self.data = {d.label: d for d in data}
        assert len(data) == len(self.data)
        self.crossings = [d for d in data if isinstance(d, Crossing)]
        self.vertices = [d for d in data if isinstance(d, Vertex)]
        self.edges = [d for d in data if isinstance(d, Edge)]

        if len(self.edges) == 0 and len(data) > 0:
            self._inflate_edges()

        if check:
            self._check()

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

    def euler(self):
        """
        Returns the Euler characteristic of the diagram.
        """
        v = len(self.crossings) + len(self.vertices)
        e = len(self.edges)
        f = len(self.faces())
        return v - e + f

    def is_planar(self):
        """
        Returns True if the diagram is planar.
        """
        return self.euler() == 2 * len(list(nx.connected_components(self.projection_graph())))

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
        assert self.is_planar()

    def _inflate_edges(self):
        """
        Creates and inserts an edge for each pair of crossings and/or vertices.
        """

        curr_index = 0
        edges = []

        for A in self.crossings + self.vertices:
            for i in range(A.degree):
                B, j = A.adjacent[i]
                if not isinstance(B, Edge):
                    E = Edge(curr_index)
                    curr_index += 1
                    edges.append(E)
                    self.data[E.label] = E
                    E[0] = (A, i)
                    E[1] = (B, j)

        self.edges = edges

    def get_object(self, label):
        return self.data[label]


    def add_vertex(self, vertex):
        """Adds a vertex to the diagram."""
        self.vertices.append(vertex)
        self.data[vertex.label] = vertex

    def remove_edge(self, edge):
        """Removes an edge from the diagram."""
        self.edges.remove(edge)
        self.data.pop(edge.label)

    def remove_crossing(self, crossing):
        """Removes a crossing from the diagram."""
        self.crossings.remove(crossing)
        self.data.pop(crossing.label)

    def connect(self, obj_1, idx, obj_2, jdx):

        both_are_edges = isinstance(obj_1, Edge) and isinstance(obj_2, Edge)

        if both_are_edges:
            V = Vertex(2, 'v' + str(len(self.vertices) + 1))
            self.add_vertex(V)
            obj_1[idx] = V[0]
            obj_2[jdx] = V[1]

        else:
            obj_1[idx] = obj_2[jdx]

    def short_cut(self, crossing, i0):
        """
        Short-cuts a crossing by removing the edge between them.
        """

        i1 = (i0 + 1) % 4
        E0, j0 = crossing.adjacent[i0]
        E1, j1 = crossing.adjacent[i1]
        if E0 == E1:
            V0 = Vertex(2, repr(E0) + '_stopper')
            self.add_vertex(V0)
            V0[0] = E0[j0]
            V0[1] = E1[j1]
        else:
            E1[j1] = E0[j0]
            E1.fuse()
            self.remove_edge(E1)

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

    def underlying_planar_embedding(self):
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


    def yamada_polynomial(self, check_pieces=False):

        A = pari('A')

        if len(self.crossings) == 0:
            return h_poly(self.projection_graph())

        C = self.crossings[0]
        c = C.label

        # S_plus
        S_plus = self.copy()
        C_plus = S_plus.data[c]
        S_plus.remove_crossing(C_plus)
        S_plus.short_cut(C_plus, 0)
        S_plus.short_cut(C_plus, 2)
        if check_pieces:
            S_plus._check()

        # S_minus
        S_minus = self.copy()
        C_minus = S_minus.data[c]
        S_minus.remove_crossing(C_minus)
        S_minus.short_cut(C_minus, 1)
        S_minus.short_cut(C_minus, 3)
        if check_pieces:
            S_minus._check()

        # S_0
        S_0 = self.copy()
        C_0 = S_0.data[c]
        S_0.remove_crossing(C_0)

        V = Vertex(4, repr(C_0) + '_smushed')
        S_0.add_vertex(V)

        for i in range(4):
            B, j = C_0.adjacent[i]
            V[i] = B[j]

        if check_pieces:
            S_0._check()

        Y_plus  = S_plus.yamada_polynomial()
        Y_minus = S_minus.yamada_polynomial()
        Y_0     = S_0.yamada_polynomial()

        return A * Y_plus + (A ** -1) * Y_minus + Y_0


    def normalized_yamada_polynomial(self):
        """normalized_yamada_polynomial"""

        yamada_polynomial = self.yamada_polynomial()

        A = pari('A')

        _, exps = get_coefficients_and_exponents(yamada_polynomial)
        a, b = min(exps), max(exps)
        ans1 = (-A) ** (-a) * yamada_polynomial
        ans2 = (-A) ** b * reverse_poly(yamada_polynomial)

        normalized_yamada_polynomial = min([ans1, ans2], key=list)

        return normalized_yamada_polynomial


def normalize_yamada_polynomial(yamada_polynomial):
    """normalized_yamada_polynomial
    """

    A = pari('A')

    _, exps = get_coefficients_and_exponents(yamada_polynomial)
    a, b = min(exps), max(exps)
    ans1 = (-A) ** (-a) * yamada_polynomial
    ans2 = (-A) ** b * reverse_poly(yamada_polynomial)

    normalized_yamada_polynomial = min([ans1, ans2], key=list)

    return normalized_yamada_polynomial

def reverse_poly(poly):
    """
    """

    A = pari('A')

    coeffs, exps = get_coefficients_and_exponents(poly)

    ans = pari(0)

    for c, e in zip(coeffs, exps):
        ans += c * A ** (-e)

    return ans


