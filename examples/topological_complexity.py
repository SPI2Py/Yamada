import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



def initial_guess_perimeter(num_interior, domain_size=1.0):
    """
    Constructs an initial guess for the interior control points along the long-way perimeter.
    The full (fixed) path is: start -> (domain_size, 0) -> (domain_size, domain_size) -> stop.
    The interior control points are evenly spaced along this polyline.
    """
    # Fixed start and stop points.
    start = np.array([0.0, 0.0])
    stop = np.array([0.0, 1.0])

    # Define the vertices of the perimeter path.
    vertices = [start, np.array([domain_size, 0.0]), np.array([domain_size, domain_size]), stop]

    # Compute cumulative distances along the polyline.
    cumulative_dist = [0]
    for i in range(len(vertices) - 1):
        seg_len = np.linalg.norm(vertices[i + 1] - vertices[i])
        cumulative_dist.append(cumulative_dist[-1] + seg_len)
    total_len = cumulative_dist[-1]

    # Determine target distances for the interior points (evenly spaced).
    targets = [(i + 1) * total_len / (num_interior + 1) for i in range(num_interior)]

    guess_points = []
    for t in targets:
        # Find the segment on which the target distance falls.
        for j in range(len(cumulative_dist) - 1):
            if cumulative_dist[j] <= t <= cumulative_dist[j + 1]:
                seg_frac = (t - cumulative_dist[j]) / (cumulative_dist[j + 1] - cumulative_dist[j])
                pt = vertices[j] + seg_frac * (vertices[j + 1] - vertices[j])
                guess_points.append(pt)
                break

    # Flatten the list of points into a single vector: [x1, y1, x2, y2, ...]
    return np.array(guess_points).flatten()


def plot_solution(opt_result, obstacles, num_interior_points, domain_size=1.0, initial_guess=None):
    """
    Plots the optimized pipe path along with obstacles.

    Parameters:
    - opt_result: the optimization result from scipy.optimize.minimize.
    - obstacles: list of obstacles as (center, radius).
    - num_interior_points: number of interior control points used.
    - domain_size: size of the square domain.
    - initial_guess: optional; the initial guess vector (flattened interior control points).
      If provided, the initial path will be plotted for comparison.
    """
    # Fixed endpoints.
    start = np.array([0.0, 0.0])
    stop = np.array([0.0, 1.0])

    # Reconstruct the optimized path.
    control_vars = opt_result.x
    pts = [start]
    for i in range(num_interior_points):
        pts.append([control_vars[2 * i], control_vars[2 * i + 1]])
    pts.append(stop)
    pts = np.array(pts)

    plt.figure(figsize=(6, 6))

    # Global Optimal Path
    plt.plot((0, 0), (0, 1), 'g-', linewidth=2, label='Global Optimized Path')

    # Plot the optimized path.
    plt.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2, label='Optimized Path')
    plt.scatter(pts[:,0], pts[:,1], color='blue', marker='o', s=50, label='Optimized Points')

    # If an initial guess is provided, reconstruct and plot it.
    if initial_guess is not None:
        guess_pts = [start]
        # Reshape the flattened vector into a list of points.
        guess_pts.extend(initial_guess.reshape(-1, 2))
        guess_pts.append(stop)
        guess_pts = np.array(guess_pts)
        plt.plot(guess_pts[:, 0], guess_pts[:, 1], 'r--', linewidth=2, label='Initial Guess')
        plt.scatter(guess_pts[:, 0], guess_pts[:, 1], color='red', marker='x', s=50, label='Initial Guess Points')
        # plt.scatter(initial_guess[::2], initial_guess[1::2], color='red', marker='x', s=50, label='Initial Guess Points')

    # Plot each obstacle as a circle.
    for center, radius in obstacles:
        circle = plt.Circle(center, radius, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.xlim(-0.1, domain_size + 0.1)
    plt.ylim(-0.1, domain_size + 0.1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pipe Routing Optimization')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show(block=True)


# Example usage:
if __name__ == "__main__":
    # Experiment parameters.
    num_interior_points = 2  # Number of control points (thus three segments).
    domain_size = 1.0
    pipe_radius = 0.02
    num_obstacles = 5


    # Generate obstacles (with a margin to avoid interfering with the initial perimeter path).
    def generate_obstacles(num_obstacles, obstacle_radius_range=(0.02, 0.05), margin=0.1):
        obstacles = []
        for _ in range(num_obstacles):
            center = [np.random.uniform(margin, domain_size - margin),
                      np.random.uniform(margin, domain_size - margin)]
            radius = np.random.uniform(*obstacle_radius_range)
            obstacles.append((center, radius))
        return obstacles


    obstacles = generate_obstacles(num_obstacles, margin=0.1)

    # Generate the initial guess.
    x0 = initial_guess_perimeter(num_interior_points, domain_size)

    # Define start and stop for reconstruction within optimization functions.
    start = np.array([0.0, 0.0])
    stop = np.array([0.0, 1.0])


    # The objective and constraints functions are assumed to be defined elsewhere.
    # For demonstration, we'll reuse the earlier definitions:
    def objective(control_vars):
        num_points = len(control_vars) // 2
        pts = [start]
        for i in range(num_points):
            pts.append([control_vars[2 * i], control_vars[2 * i + 1]])
        pts.append(stop)
        return sum(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i])) for i in range(len(pts) - 1))


    def signed_distance_segment_circle(p1, p2, circle_center, circle_radius):
        p1 = np.array(p1)
        p2 = np.array(p2)
        circle_center = np.array(circle_center)
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len == 0:
            return np.linalg.norm(circle_center - p1) - circle_radius
        seg_dir = seg_vec / seg_len
        proj = np.dot(circle_center - p1, seg_dir)
        if proj < 0:
            closest = p1
        elif proj > seg_len:
            closest = p2
        else:
            closest = p1 + proj * seg_dir
        return np.linalg.norm(circle_center - closest) - circle_radius


    def collision_constraints(control_vars, obstacles, pipe_radius):
        num_points = len(control_vars) // 2
        pts = [start]
        for i in range(num_points):
            pts.append([control_vars[2 * i], control_vars[2 * i + 1]])
        pts.append(stop)
        cons = []
        for (center, radius) in obstacles:
            for i in range(len(pts) - 1):
                dist = signed_distance_segment_circle(pts[i], pts[i + 1], center, radius)
                cons.append(dist - pipe_radius)
        return np.array(cons)


    def constraint_wrapper(control_vars, obstacles, pipe_radius):
        return collision_constraints(control_vars, obstacles, pipe_radius)


    # Setup constraints for SLSQP.
    cons = {
        'type': 'ineq',
        'fun': lambda vars: constraint_wrapper(vars, obstacles, pipe_radius)
    }

    # Run the optimization.
    from scipy.optimize import minimize

    res = minimize(objective, x0, method='SLSQP', constraints=cons,
                   bounds=[(0, domain_size)] * (num_interior_points * 2),
                   options={'disp': True, 'maxiter': 500})

    # Plot the results.
    plot_solution(res, obstacles, num_interior_points, domain_size, initial_guess=x0)
