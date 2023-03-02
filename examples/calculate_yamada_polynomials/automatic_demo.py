import numpy as np
from yamada.projection import SpatialGraph

for i in range(6):
    np.random.seed(i)

    # Random seed 0 works
    # Random seed 1 does not work

    # sg1 = SpatialGraph(nodes=['a', 'b', 'c', 'd'],
    #                    node_positions=np.array([[0, 0.5, 0], [1, 0.5, 1], [1, 0, 0], [0, 0, 1]]),
    #                    edges=[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')])
    # sg1.project()
    # sg1.plot()
    # sgd1 = sg1.create_spatial_graph_diagram()
    # yamada_polynomial_infinity_symbol = sgd1.yamada_polynomial()
    # print("Infinity Symbol Yamada Polynomial:", yamada_polynomial_infinity_symbol)
    #
    # node_list = sg1.get_adjacent_nodes('a')
    #
    # node_ordering = sg1.cyclic_edge_order_vertex('a')
    #
    # crossing_ordering = sg1.cyclic_edge_order_crossing(0)

    # node_pairs = sg1.get_adjacent_edges_without_crossings('a')

    # sg1 = SpatialGraph(nodes=['a', 'b', 'c', 'd', 'e', 'f'],
    #                    node_positions=np.array([[0, 0.5, 0], [1, 0, 1], [2, 0.5, 0], [3, 0, 1], [1, 1, 0], [-1, 0, 1]]),
    #                    edges=[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'a')])
    #
    # sg1.project()
    # sg1.plot()
    # sgd1 = sg1.create_spatial_graph_diagram()
    # yamada_polynomial_infinity_symbol = sgd1.yamada_polynomial()
    # print("Infinity Symbol Yamada Polynomial:", yamada_polynomial_infinity_symbol)


    sg1 = SpatialGraph(nodes=['a', 'b', 'c', 'd', 'e', 'f'],
                       node_positions=np.array([[0, 0.5, 0], [1, 0, 1], [2, 0.5, 0], [3, 0, 1], [1, 1, 0], [-1, 0, 1]]),
                       edges=[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'a'), ('b', 'f')])

    sg1.project()
    sg1.plot()


    sgd1 = sg1.create_spatial_graph_diagram()

    yamada_polynomial_infinity_symbol = sgd1.yamada_polynomial()
    #
    print("Infinity Symbol Yamada Polynomial:", yamada_polynomial_infinity_symbol)
