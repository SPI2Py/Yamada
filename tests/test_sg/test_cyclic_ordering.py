import numpy as np
from yamada import SpatialGraph


def test_cyclic_node_ordering_vertex():
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    node_positions = {'a': [0, 0, 0], 'b': [1, 0, 0], 'c': [0.5, 1, 0], 'd': [0.5, 0.5, 0], 'e': [0.25, 0.75, 0],
                      'f': [0.75, 0.75, 0], 'g': [0, 1, 0], 'h': [1, 1, 0]}

    edges = [('a', 'b'), ('a', 'g'), ('a', 'd'), ('b', 'd'), ('b', 'h'), ('d', 'e'), ('d', 'f'), ('e', 'c'), ('f', 'c'),
             ('g', 'c'), ('h', 'c')]

    # Use a predefined rotation (from a random seed) that previously produced an error
    rotation = np.array([3.44829694, 4.49366732, 3.78727399])

    sg = SpatialGraph(nodes=nodes,
                      node_positions=node_positions,
                      edges=edges,
                      rotation=rotation)

    order = sg.cyclic_order_vertex('c')
    expected_order = {'c': {'e': 3, 'f': 0, 'g': 2, 'h': 1}}

    assert order == expected_order


def test_cyclic_ordering_crossing():
    # TODO Re-label these! Or Find out what the actual rotations are and update them!
    component_a = 'comp_a'
    component_b = 'comp_b'
    component_c = 'comp_c'
    component_d = 'comp_d'
    component_e = 'comp_e'
    component_f = 'comp_f'
    component_g = 'comp_g'
    component_h = 'comp_h'

    waypoint_ab = 'w_ab'
    waypoint_ad = 'w_ad'
    waypoint_ae = 'w_ae'
    waypoint_bc = 'w_bc'
    waypoint_bf = 'w_bf'
    waypoint_cd = 'w_cd'
    waypoint_cg = 'w_cg'
    waypoint_dh = 'w_dh'
    waypoint_ef = 'w_ef'
    waypoint_eh = 'w_eh'
    waypoint_fg = 'w_fg'
    waypoint_gh = 'w_gh'

    nodes = [component_a, component_b, component_c, component_d, component_e, component_f,
             component_g, component_h, waypoint_ab, waypoint_ad, waypoint_ae, waypoint_bc,
             waypoint_bf, waypoint_cd, waypoint_cg, waypoint_dh, waypoint_ef, waypoint_eh,
             waypoint_fg, waypoint_gh]

    component_positions = np.array([[0, 0, 0],  # a
                                [1, 0, 0],  # b
                                [1, 1, 0],  # c
                                [0, 1, 0],  # d
                                [0, 0, 1],  # e
                                [1, 0, 1],  # f
                                [1, 1, 1],  # g
                                [0, 1, 1]])  # h

    waypoint_positions = np.array([[0.5, 0, 0],  # ab
                               [0, 0.5, 0],  # ad
                               [0, 0, 0.5],  # ae
                               [1, 0.5, 0],  # bc
                               [1, 0, 0.5],  # bf
                               [0.5, 1, 0],  # cd
                               [1, 1, 0.5],  # cg
                               [0, 1, 0.5],  # dh
                               [0.5, 0, 1],  # ef
                               [0, 0.5, 1],  # eh
                               [1, 0.5, 1],  # fg
                               [0.5, 1, 1]])  # gh

    node_positions = np.concatenate((component_positions, waypoint_positions), axis=0)

    node_positions = {node: pos for node, pos in zip(nodes, node_positions)}

    edges = [(component_a, waypoint_ab), (waypoint_ab, component_b),
         (component_a, waypoint_ad), (waypoint_ad, component_d),
         (component_a, waypoint_ae), (waypoint_ae, component_e),
         (component_b, waypoint_bc), (waypoint_bc, component_c),
         (component_b, waypoint_bf), (waypoint_bf, component_f),
         (component_c, waypoint_cd), (waypoint_cd, component_d),
         (component_c, waypoint_cg), (waypoint_cg, component_g),
         (component_d, waypoint_dh), (waypoint_dh, component_h),
         (component_e, waypoint_ef), (waypoint_ef, component_f),
         (component_e, waypoint_eh), (waypoint_eh, component_h),
         (component_f, waypoint_fg), (waypoint_fg, component_g),
         (component_g, waypoint_gh), (waypoint_gh, component_h)]

    # Define the random rotation that previously caused issues
    rotation = np.array([3.44829694, 4.49366732, 3.78727399])

    sg = SpatialGraph(nodes=nodes,
                      node_positions=node_positions,
                      edges=edges,
                      rotation=rotation)

    ordering_dict = sg.cyclic_order_crossings()

    expected_dict = {'crossing_0': {'comp_c': 2, 'w_ef': 3, 'w_bc': 0, 'comp_f': 1},
                     'crossing_1': {'w_cd': 0, 'w_eh': 1, 'comp_d': 2, 'comp_e': 3}}

    assert ordering_dict == expected_dict


def test_cyclic_ordering_crossing_2():
    component_a = 'comp_a'
    component_b = 'comp_b'
    component_c = 'comp_c'
    component_d = 'comp_d'
    component_e = 'comp_e'
    component_f = 'comp_f'
    component_g = 'comp_g'
    component_h = 'comp_h'

    waypoint_ab = 'w_ab'
    waypoint_ad = 'w_ad'
    waypoint_ae = 'w_ae'
    waypoint_bc = 'w_bc'
    waypoint_bf = 'w_bf'
    waypoint_cd = 'w_cd'
    waypoint_cg = 'w_cg'
    waypoint_dh = 'w_dh'
    waypoint_ef = 'w_ef'
    waypoint_eh = 'w_eh'
    waypoint_fg = 'w_fg'
    waypoint_gh = 'w_gh'

    nodes = [component_a, component_b, component_c, component_d, component_e, component_f,
             component_g, component_h, waypoint_ab, waypoint_ad, waypoint_ae, waypoint_bc,
             waypoint_bf, waypoint_cd, waypoint_cg, waypoint_dh, waypoint_ef, waypoint_eh,
             waypoint_fg, waypoint_gh]

    component_positions = np.array([[0, 0, 0],  # a
                                    [1, 0, 0],  # b
                                    [1, 1, 0],  # c
                                    [0, 1, 0],  # d
                                    [0, 0, 1],  # e
                                    [1, 0, 1],  # f
                                    [1, 1, 1],  # g
                                    [0, 1, 1]])  # h

    waypoint_positions = np.array([[0.5, 0.1, 0],  # ab
                                   [0.1, 0.7, 0.2],  # ad
                                   [0.1, 0, 0.5],  # ae
                                   [1, 0.5, 0],  # bc
                                   [1, 0.1, 0.5],  # bf
                                   [0.5, 1, 0],  # cd
                                   [0.7, 0.6, 0.5],  # cg
                                   [0.1, 1, 0.5],  # dh
                                   [0.5, 0.1, 1],  # ef
                                   [0.1, 0.6, 1],  # eh
                                   [1, 0.5, 1],  # fg
                                   [0.5, 0.95, 1]])  # gh

    node_positions = np.concatenate((component_positions, waypoint_positions), axis=0)

    node_positions = {node: pos for node, pos in zip(nodes, node_positions)}

    edges = [(component_a, waypoint_ab), (waypoint_ab, component_b),
             (component_a, waypoint_ad), (waypoint_ad, component_d),
             (component_a, waypoint_ae), (waypoint_ae, component_e),
             (component_b, waypoint_bc), (waypoint_bc, component_c),
             (component_b, waypoint_bf), (waypoint_bf, component_f),
             (component_c, waypoint_cd), (waypoint_cd, component_d),
             (component_c, waypoint_cg), (waypoint_cg, component_g),
             (component_d, waypoint_dh), (waypoint_dh, component_h),
             (component_e, waypoint_ef), (waypoint_ef, component_f),
             (component_e, waypoint_eh), (waypoint_eh, component_h),
             (component_f, waypoint_fg), (waypoint_fg, component_g),
             (component_g, waypoint_gh), (waypoint_gh, component_h)]

    # Set rotation
    rotation = np.array([2.73943676, 0.16289932, 3.4536312])

    sg = SpatialGraph(nodes=nodes,
                      node_positions=node_positions,
                      edges=edges,
                      rotation=rotation)

    ordering_dict = sg.cyclic_order_crossings()

    expected_dict = {'crossing_0': {'comp_a': 1, 'comp_d': 2, 'w_ab': 3, 'w_dh': 0},
                     'crossing_1': {'comp_b': 1, 'comp_c': 2, 'crossing_2': 3, 'w_cg': 0},
                     'crossing_2': {'w_cg': 0, 'crossing_1': 1, 'comp_g': 2, 'w_bf': 3},
                     'crossing_3': {'w_gh': 0, 'w_bf': 1, 'comp_g': 2, 'comp_f': 3}}

    assert ordering_dict == expected_dict