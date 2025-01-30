"""A module to computer to H polynomial of an Abstract Graph

Computation of the H polynomial is based on the following paper:

SYSTEMATIC ENUMERATION AND IDENTIFICATION OF UNIQUE SPATIAL TOPOLOGIES OF 3D SYSTEMS USING SPATIAL GRAPH REPRESENTATIONS
Peddada et al.
DOI: 10.1115/DETC2021-66900

P1. H(empty graph) = 1 and H(simple loop) = A + 1 + A^(-1)
P2. H(G_1 disjoint union G_2)      =  H(G_1) * H(G_2)
P3. H(G_1 union G_2)               = -H(G_1) * H(G_2)
P4. If G has a cut edge, then H(G) =  0
P5. Let e be a non-loop edge of graph G. Then H(G) = H(G/e) + H(G-e) where G/e is the graph obtained from G by
    contracting e to a point and G-e is G with e deleted.

"""

import networkx as nx
import collections
from cypari import pari


def has_cut_edge(abstract_graph):

    """
    Determines if the abstract graph has a cut edge. A cut edge is an edge that, if removed, would increase the number
    of disconnected sub-graphs in the graph.
    """

    g = nx.Graph(abstract_graph)

    for u, v in nx.bridges(g):
        if abstract_graph.number_of_edges(u, v) == 1:
            return True

    return False


def an_edge(graph):

    """
    A generator that returns the next edge in the graph.
    """

    return next(iter(graph.edges()))


def remove_valence_two_vertices(graph):

    """
    TODO Why this?
    """

    G = graph.copy()
    valence_two = {v for v, d in G.degree if d == 2}
    T = G.subgraph(valence_two)
    new_edges = []
    for comp in nx.connected_components(T):
        C = T.subgraph(comp)
        if C.number_of_nodes() == C.number_of_edges():
            x = next(iter(comp))
            y = x
        else:
            if len(comp) == 1:
                c = next(iter(comp))
                x, y = [e[1] for e in G.edges(c)]
            else:
                a, b = [c for c, d in C.degree if d == 1]
                x = [e[1] for e in G.edges(a) if e[1] not in comp][0]
                y = [e[1] for e in G.edges(b) if e[1] not in comp][0]
        new_edges.append((x, y))
    G.remove_nodes_from(valence_two)
    G.add_edges_from(new_edges)
    return G


def graph_hash(graph):

    """
    Returns a hash of the graph.

    The graph hash is used to cache the results of the H polynomial computation and avoid recomputing the H polynomial.
    """

    if isinstance(graph, nx.MultiGraph):
        # Convert MultiGraph to a simple Graph (ignoring multiple edges)
        graph = nx.Graph(graph)

    return nx.weisfeiler_lehman_graph_hash(graph, iterations=3)


H_poly_cache = collections.defaultdict(list)


def h_poly(g):

    """
    Computes the H polynomial of the abstract graph g.

    :param g: The abstract graph
    :return: The H polynomial of the abstract graph g
    """

    # Define the indeterminate A
    a = pari('A')

    # Validate the input format
    if not isinstance(g, nx.MultiGraph):
        g = nx.MultiGraph(g)


    # RECURSION BASE CONDITION
    # Abstract graph is empty: H Polynomial property P1: H(empty graph) = 1


    if g.number_of_nodes() == 0:
        return pari(1)


    # RECURSION LOGIC
    # Case 1: If all nodes of the graph are connected then recursively calculate the H polynomial of the graph.
    # Case 2: Else, the abstract graph contains subgraphs that are not connected. Recursively calculate the H polynomial
    # of each subgraph (see "if statement" above).

    # Recursion logic case 1: The abstract graph is connected.
    if nx.is_connected(g):

        # Check if the graph has already been computed
        its_hash = graph_hash(g)
        for H, poly in H_poly_cache[its_hash]:
            if nx.is_isomorphic(g, H):
                return poly
        g_for_cache = g.copy()

        # H Polynomial property P4
        # If G has a cut edge, then H(G) =  0
        if has_cut_edge(g):
            ans = pari(0)

        else:
            # TODO Why remove valence two vertices? Finish logic here...
            # Collapse valence two vertices (b/c do not affect calc...)
            g = remove_valence_two_vertices(g)

            loop_factor = pari(1)

            # P1 & P3

            loops = [e for e in g.edges() if e[0] == e[1]]

            for u, v in loops:
                g.remove_edge(u, v)
                loop_factor = -loop_factor * (a + 1 + a ** -1)

            # correcting for the difference between ...
            if g.number_of_nodes() == 1:
                ans = -loop_factor

            # TBD (check out the paper for formula, avoids P5 recursion?)
            elif g.number_of_nodes() == 2:

                b = -(a ** -1 + 2 + a)
                q = g.number_of_edges()

                h = ((b + 1) - (b + 1) ** q) / b

                ans = loop_factor * h

            # H Polynomial property P5.
            # H(G) = H(G/e) + H(G-e)
            else:
                e = an_edge(g)
                g_mod_e = nx.contracted_edge(g, e, self_loops=True)
                g_mod_e.remove_edge(e[0], e[0])
                g.remove_edge(*e)
                ans = (h_poly(g_mod_e) + h_poly(g)) * loop_factor

        H_poly_cache[its_hash].append((g_for_cache, ans))
        return ans

    # Recursion logic case 2: The abstract graph contains subgraphs that are not connected.
    else:

        # Must initialize ans as 1 for recursive multiplication.
        ans = pari(1)

        # H polynomial property P2
        # H(G_1 disjoint union G_2) = H(G_1) * H(G_2) where H(G_1) and H(G_2) are recursively computed.
        for vertices in nx.connected_components(g):
            S = g.subgraph(vertices).copy()
            h = h_poly(S)
            if h == 0:
                return pari(0)

            ans = ans * h

        return ans