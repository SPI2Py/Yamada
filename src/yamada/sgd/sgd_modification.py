def split_edges(edges):
    """
    Splits a list of edges with more than two nodes into a series of edges which each have two nodes.
    """

    # Strip crossings from edges
    edges_without_crossings = []
    for edge in edges:
        edge_without_crossings = [node for node in edge if "C" not in node]
        edges_without_crossings.append(edge_without_crossings)

    new_edges = []
    for edge in edges_without_crossings:
        new_edges.extend(split_edge(edge))
    return new_edges


def split_edge(edge):
    """
    Splits an edge with more than two nodes into a series of edges which each have two nodes.
    """

    edges = []
    for i in range(len(edge)-1):
        edges.append((edge[i], edge[i+1]))
    return edges


def apply_crossing_swap(sgd, crossing_label):

    sgd = sgd.copy()

    X = sgd.get_object(crossing_label)

    (A, i), (B, j), (C, k), (D, l) = X.adjacent

    # Swap the under-crossing strand and over-crossing strand
    X[0] = D[l]
    X[1] = A[i]
    X[2] = B[j]
    X[3] = C[k]

    return sgd
