import numpy as np
from yamada import SpatialGraph
import matplotlib.pyplot as plt

# np.random.seed(0)
np.random.seed(0)
# There should be a crossing... for 2
# why does 6 break it?

component_a = 'c_a'
component_b = 'c_b'
component_c = 'c_c'
component_d = 'c_d'
component_e = 'c_e'
component_f = 'c_f'
component_g = 'c_g'
component_h = 'c_h'

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

nodes = [component_a, component_b, component_c, component_d, component_e, component_f, component_g, component_h,
         waypoint_ab, waypoint_ad, waypoint_ae, waypoint_bc, waypoint_bf, waypoint_cd, waypoint_cg, waypoint_dh,
         waypoint_ef, waypoint_eh, waypoint_fg, waypoint_gh]

# nodes = [component_a, component_b, component_c, component_d, component_e, component_f, component_g, component_h]

component_positions = np.array([[0, 0, 0],  # a
                                [1, 0, 0],  # b
                                [1, 1, 0],  # c
                                [0, 1, 0],  # d
                                [0, 0, 1],  # e
                                [1, 0, 1],  # f
                                [1, 1, 1],  # g
                                [0, 1, 1]])  # h



# waypoint_positions = np.random.rand(12, 3)
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

# waypoint_positions = np.array([[0.5, 0.1, 0],  # ab
#                                [0.4, 0.7, 0.3],  # ad
#                                [0.1, 0, 0.5],  # ae
#                                [1, 0.5, 0],  # bc
#                                [1, 0.1, 0.5],  # bf
#                                [0.5, 1, 0],  # cd
#                                [0.4, 0.6, 0.5],  # cg
#                                [0.2, 1, 0.5],  # dh
#                                [0.5, 0.1, 1],  # ef
#                                [0.1, 0.7, 1],  # eh
#                                [1, 0.5, 1],  # fg
#                                [0.5, 0.85, 1]])  # gh


# node_positions = component_positions
node_positions = np.concatenate((component_positions, waypoint_positions), axis=0)


# edges = [(component_a, component_b),
#          (component_a, component_d),
#          (component_a, component_e),
#          (component_b, component_c),
#          (component_b, component_f),
#          (component_c, component_d),
#          (component_c, component_g),
#          (component_d, component_h),
#          (component_e, component_f),
#          (component_e, component_h),
#          (component_f, component_g),
#          (component_g, component_h)]

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





sg = SpatialGraph(nodes=nodes, node_positions=list(node_positions), edges=edges)
sg.plot()

sgd = sg.create_spatial_graph_diagram()
yp = sgd.normalized_yamada_polynomial()

print("Yamada Polynomial:", yp)

# sg = SpatialGraph(nodes=['a', 'b', 'c', 'd'],
#                       node_positions=np.array([[0, 0.5, 0], [1, 0.5, 1], [1, 0, 0], [0, 0, 1]]),
#                       edges=[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')])
#
# sep = sg.get_sub_edges()
#
# expected_sub_edges = [('b', 'crossing_0'), ('crossing_0', 'a'), ('b', 'c'), ('d', 'crossing_0'), ('crossing_0', 'c'), ('d', 'a')]
#
# print('ans: ', sep)
# print('exp: ', expected_sub_edges)

# Random seed 0
# For get crossings
# Crossing 1  waypoint cd to component 2 (in front) X component e to waypoint eh (behind)
# Crossing 2 waypoint bc to component c (in front) X waypoint ef to component f (behind)
