from yamada.sgd.diagram_elements import Edge, Vertex, Crossing
from yamada.sgd.spatial_graph_diagrams import SpatialGraphDiagram
from yamada.sgd.topological_distance import compute_min_distance

def create_unknot():
    e1 = Edge('e1')
    v1 = Vertex(2, 'v1')
    e1[0] = v1[0]
    e1[1] = v1[1]
    return SpatialGraphDiagram(edges=[e1], vertices=[v1])


def create_figure_eight():
    e1 = Edge('e1')
    e2 = Edge('e2')
    e3 = Edge('e3')
    e4 = Edge('e4')
    e5 = Edge('e5')
    e6 = Edge('e6')

    c1 = Crossing('c1')
    c2 = Crossing('c2')
    c3 = Crossing('c3')

    c1[0] = e2[1]
    c1[1] = e1[0]
    c1[2] = e4[0]
    c1[3] = e3[1]

    c2[0] = e6[0]
    c2[1] = e2[0]
    c2[2] = e3[0]
    c2[3] = e5[0]

    c3[0] = e5[1]
    c3[1] = e4[1]
    c3[2] = e1[1]
    c3[3] = e6[1]

    return SpatialGraphDiagram(edges=[e1, e2, e3, e4, e5, e6], crossings=[c1, c2, c3])

unknot = create_unknot()
figure_eight = create_figure_eight()

distance, network = compute_min_distance(unknot, figure_eight)