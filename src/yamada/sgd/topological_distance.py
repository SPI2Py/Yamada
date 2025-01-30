from collections import deque, defaultdict
import time
from yamada.sgd.sgd_modification import apply_crossing_swap


def compute_min_distance(diagram1, diagram2, max_depth=3, max_runtime=10):
    start_time = time.time()

    # Initialize BFS queues and visited polynomial sets
    queue1 = deque([(diagram1, 0)])
    queue2 = deque([(diagram2, 0)])
    polynomials1 = {diagram1.yamada_polynomial(): 0}
    polynomials2 = {diagram2.yamada_polynomial(): 0}
    network = defaultdict(list) # Stores connections between topologies

    # Quick check if already matching
    if diagram1.yamada_polynomial() == diagram2.yamada_polynomial():
        return 0, network

    while queue1 or queue2:
        # Time check
        if time.time() - start_time > max_runtime:
            print("Time limit reached")
            return None, network

        # Expand BFS from diagram1
        if queue1:
            current_diagram1, depth1 = queue1.popleft()
            if depth1 < max_depth:
                for crossing in current_diagram1.crossings:
                    new_diagram = apply_crossing_swap(current_diagram1, crossing.label)
                    yamada_poly_new = new_diagram.yamada_polynomial()
                    yamada_poly_current = current_diagram1.yamada_polynomial()
                    if yamada_poly_new not in polynomials1:

                        # Add to network
                        network[yamada_poly_current].append(yamada_poly_new)

                        # Update BFS structures
                        polynomials1[yamada_poly_new] = depth1 + 1
                        queue1.append((new_diagram, depth1 + 1))

                        # EARLY EXIT on match
                        if yamada_poly_new in polynomials2:
                            return depth1 + 1 + polynomials2[yamada_poly_new], network

        # Expand BFS from diagram2
        if queue2:
            current_diagram2, depth2 = queue2.popleft()
            if depth2 < max_depth:
                for crossing in current_diagram2.crossings:
                    new_diagram = apply_crossing_swap(current_diagram2, crossing.label)
                    yamada_poly_new = new_diagram.yamada_polynomial()
                    yamada_poly_current = current_diagram2.yamada_polynomial()
                    if yamada_poly_new not in polynomials2:

                        # Add to network
                        network[yamada_poly_current].append(yamada_poly_new)

                        # Update BFS structures
                        polynomials2[yamada_poly_new] = depth2 + 1
                        queue2.append((new_diagram, depth2 + 1))

                        # EARLY EXIT on match
                        if yamada_poly_new in polynomials1:
                            return depth2 + 1 + polynomials1[yamada_poly_new], network

    # No match found
    return None, network


# def compute_min_distance(diagram1, diagram2, max_depth=3, max_runtime=10):
#     start_time = time.time()
#
#     # Initialize BFS queues and visited polynomial sets
#     queue1 = deque([(diagram1, 0)])
#     queue2 = deque([(diagram2, 0)])
#     polynomials1 = {diagram1.yamada_polynomial(): 0}
#     polynomials2 = {diagram2.yamada_polynomial(): 0}
#
#     # Quick check if already matching
#     if diagram1.yamada_polynomial() == diagram2.yamada_polynomial():
#         return 0
#
#     while queue1 or queue2:
#         # Time check
#         if time.time() - start_time > max_runtime:
#             print("Time limit reached")
#             return None
#
#         # Expand BFS from diagram1
#         if queue1:
#             current_diagram1, depth1 = queue1.popleft()
#             if depth1 < max_depth:
#                 for crossing in current_diagram1.crossings:
#                     new_diagram = apply_crossing_swap(current_diagram1, crossing.label)
#                     yamada_poly = new_diagram.yamada_polynomial()
#                     if yamada_poly not in polynomials1:
#                         polynomials1[yamada_poly] = depth1 + 1
#                         queue1.append((new_diagram, depth1 + 1))
#
#                         # EARLY EXIT on match
#                         if yamada_poly in polynomials2:
#                             return depth1 + 1 + polynomials2[yamada_poly]
#
#         # Expand BFS from diagram2
#         if queue2:
#             current_diagram2, depth2 = queue2.popleft()
#             if depth2 < max_depth:
#                 for crossing in current_diagram2.crossings:
#                     new_diagram = apply_crossing_swap(current_diagram2, crossing.label)
#                     yamada_poly = new_diagram.yamada_polynomial()
#                     if yamada_poly not in polynomials2:
#                         polynomials2[yamada_poly] = depth2 + 1
#                         queue2.append((new_diagram, depth2 + 1))
#
#                         # EARLY EXIT on match
#                         if yamada_poly in polynomials1:
#                             return depth2 + 1 + polynomials1[yamada_poly]
#
#     # No match found
#     return None
