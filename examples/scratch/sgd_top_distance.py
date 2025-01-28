from cypari import pari
from yamada import Edge, Crossing, Vertex, SpatialGraphDiagram
from yamada.sgd.sgd_analysis import available_crossing_swaps
from yamada.sgd.sgd_modification import apply_crossing_swap
from yamada.sgd.topological_distance import compute_min_distance

e1 = Edge('e1')
e2 = Edge('e2')
e3 = Edge('e3')
e4 = Edge('e4')
e5 = Edge('e5')
e6 = Edge('e6')
e7 = Edge('e7')
e8 = Edge('e8')
e9 = Edge('e9')
e10 = Edge('e10')
e11 = Edge('e11')
e12 = Edge('e12')
v1 = Vertex(3, 'v1')
v2 = Vertex(3, 'v2')
v3 = Vertex(3, 'v3')
v4 = Vertex(3, 'v4')
c1 = Crossing('c1')
c2 = Crossing('c2')
c3 = Crossing('c3')

c1[0] = e12[0]
c1[1] = e11[0]
c1[2] = e5[1]
c1[3] = e6[1]

c2[0] = e10[1]
c2[1] = e9[1]
c2[2] = e11[1]
c2[3] = e12[1]

c3[0] = e7[1]
c3[1] = e8[1]
c3[2] = e9[0]
c3[3] = e10[0]

v1[0] = e4[1]
v1[1] = e1[0]
v1[2] = e5[0]

v2[0] = e6[0]
v2[1] = e1[1]
v2[2] = e2[0]

v3[0] = e3[0]
v3[1] = e7[0]
v3[2] = e2[1]

v4[0] = e4[0]
v4[1] = e8[0]
v4[2] = e3[1]

sgd = SpatialGraphDiagram(edges=[e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12],
                          vertices=[v1, v2, v3, v4],
                          crossings=[c1, c2, c3])

sgd.plot()