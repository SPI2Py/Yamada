from collections import deque, defaultdict
import time
from yamada.sgd.sgd_modification import apply_crossing_swap


def compute_min_distance(diagram1, diagram2, max_depth=5, max_runtime=10):
    """
    Perform bi-directional BFS to compute the minimum number of topological changes.
    """
    start_time = time.time()

    # Initialize BFS queues and visited sets
    queue1 = deque([(diagram1, 0)])
    queue2 = deque([(diagram2, 0)])
    polynomials1 = {diagram1.yamada_polynomial(): 0}
    polynomials2 = {diagram2.yamada_polynomial(): 0}
    network = defaultdict(list)

    # Quick check if diagrams already match
    if diagram1.yamada_polynomial() == diagram2.yamada_polynomial():
        return 0, network

    while queue1 and queue2:
        # Time check
        if time.time() - start_time > max_runtime:
            raise TimeoutError("Time limit reached")

        # Alternate BFS expansions between both sides
        for queue, polynomials, other_polynomials in [
            (queue1, polynomials1, polynomials2),
            (queue2, polynomials2, polynomials1),
        ]:
            # Expand BFS
            if queue:
                current_diagram, depth = queue.popleft()
                if depth < max_depth:
                    for crossing in current_diagram.crossings:
                        new_diagram = apply_crossing_swap(current_diagram, crossing.label)
                        yamada_poly_new = new_diagram.yamada_polynomial()
                        yamada_poly_current = current_diagram.yamada_polynomial()

                        if yamada_poly_new not in polynomials:
                            # Add to network
                            network[yamada_poly_current].append(yamada_poly_new)

                            # Update BFS structures
                            polynomials[yamada_poly_new] = depth + 1
                            queue.append((new_diagram, depth + 1))

                            # Check for early exit
                            if yamada_poly_new in other_polynomials:
                                return depth + 1 + other_polynomials[yamada_poly_new], network

    # No match found
    return None, network




# def expand_bfs(queue, polynomials, other_polynomials, network, max_depth):
#     """
#     Expand the BFS for the current diagram.
#
#     Args:
#         queue: The BFS queue containing diagrams and depths.
#         polynomials: The visited set for the current diagram.
#         other_polynomials: The visited set for the opposing diagram.
#         network: The topological network being constructed.
#         max_depth: The maximum depth for BFS expansion.
#
#     Returns:
#         The distance if an early exit is found, or None if expansion continues.
#     """
#     if not queue:
#         return None
#
#     current_diagram, depth = queue.popleft()
#     if depth >= max_depth:
#         return None
#
#     # Expand neighbors
#     for crossing in current_diagram.crossings:
#         new_diagram = apply_crossing_swap(current_diagram, crossing.label)
#         yamada_poly_new = new_diagram.yamada_polynomial()
#         yamada_poly_current = current_diagram.yamada_polynomial()
#
#         if yamada_poly_new not in polynomials:
#             # Add to network
#             network[yamada_poly_current].append(yamada_poly_new)
#
#             # Update BFS structures
#             polynomials[yamada_poly_new] = depth + 1
#             queue.append((new_diagram, depth + 1))
#
#             # Early exit if the polynomial matches one from the other BFS
#             if yamada_poly_new in other_polynomials:
#                 return depth + 1 + other_polynomials[yamada_poly_new]
#
#     return None
#
#
# def compute_min_distance(diagram1, diagram2, max_depth=3, max_runtime=10):
#     """
#     Compute the minimum number of topological changes between two diagrams.
#     """
#     start_time = time.time()
#
#     # Initialize BFS queues and visited polynomial sets
#     queue1 = deque([(diagram1, 0)])
#     queue2 = deque([(diagram2, 0)])
#     polynomials1 = {diagram1.yamada_polynomial(): 0}
#     polynomials2 = {diagram2.yamada_polynomial(): 0}
#     network = defaultdict(list)
#
#     # Quick check if already matching
#     if diagram1.yamada_polynomial() == diagram2.yamada_polynomial():
#         return 0, network
#
#     # BFS search loop
#     while queue1 or queue2:
#         # Time check
#         if time.time() - start_time > max_runtime:
#             print("Time limit reached")
#             return None, network
#
#         # Expand BFS from diagram1
#         result = expand_bfs(queue1, polynomials1, polynomials2, network, max_depth)
#         if result is not None:
#             return result, network
#
#         # Expand BFS from diagram2
#         result = expand_bfs(queue2, polynomials2, polynomials1, network, max_depth)
#         if result is not None:
#             return result, network
#
#     # No match found
#     return None, network

