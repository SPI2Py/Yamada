import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddatas
import matplotlib
matplotlib.use('TkAgg')

# Fixed endpoints and domain parameters.
start = np.array([0.0, 0.0])
stop = np.array([0.0, 1.0])
domain_size = 1.0
pipe_radius = 0.02  # clearance radius for the pipe


# --- Objective and Constraint Functions ---

def objective(control_vars):
    """
    Computes the total pipe length given the flattened interior control points.
    The full path is: start -> interior control points -> stop.
    """
    num_points = len(control_vars) // 2
    pts = [start]
    for i in range(num_points):
        pts.append([control_vars[2 * i], control_vars[2 * i + 1]])
    pts.append(stop)
    return sum(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i])) for i in range(len(pts) - 1))


def signed_distance_segment_circle(p1, p2, circle_center, circle_radius):
    """
    Computes the signed distance between the line segment (p1 to p2) and a circle.
    Returns positive if the clearance is above zero, negative if there is interference.
    """
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
    """
    Computes constraints for each segment-obstacle pair.
    Each constraint is: (signed distance) - pipe_radius >= 0.
    """
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


# --- Obstacle Generation Function ---
def generate_obstacles_coverage(coverage, obstacle_radius_range=(0.02, 0.05), domain_size=1.0, margin=0.1):
    """
    Generates obstacles until the total area of obstacles is at least 'coverage' * (domain_area).
    Obstacles are generated within [margin, domain_size - margin] to avoid the initial perimeter path.
    """
    obstacles = []
    target_area = coverage * (domain_size ** 2)
    current_area = 0.0
    max_iter = 1000
    iter_count = 0
    while current_area < target_area and iter_count < max_iter:
        center = [np.random.uniform(margin, domain_size - margin),
                  np.random.uniform(margin, domain_size - margin)]
        radius = np.random.uniform(*obstacle_radius_range)
        obstacles.append((center, radius))
        current_area += np.pi * radius ** 2
        iter_count += 1
    return obstacles


# --- Initial Guess Function ---
def initial_guess_perimeter(num_interior, domain_size=1.0):
    """
    Constructs an initial guess for the interior control points along the long way around the perimeter.
    The full path is: start -> (domain_size, 0) -> (domain_size, domain_size) -> stop.
    The interior points are evenly spaced along this polyline.
    """
    start = np.array([0.0, 0.0])
    stop = np.array([0.0, 1.0])
    vertices = [start, np.array([domain_size, 0.0]), np.array([domain_size, domain_size]), stop]

    cumulative_dist = [0]
    for i in range(len(vertices) - 1):
        seg_len = np.linalg.norm(vertices[i + 1] - vertices[i])
        cumulative_dist.append(cumulative_dist[-1] + seg_len)
    total_len = cumulative_dist[-1]

    targets = [(i + 1) * total_len / (num_interior + 1) for i in range(num_interior)]
    guess_points = []
    for t in targets:
        for j in range(len(cumulative_dist) - 1):
            if cumulative_dist[j] <= t <= cumulative_dist[j + 1]:
                seg_frac = (t - cumulative_dist[j]) / (cumulative_dist[j + 1] - cumulative_dist[j])
                pt = vertices[j] + seg_frac * (vertices[j + 1] - vertices[j])
                guess_points.append(pt)
                break
    return np.array(guess_points).flatten()


# --- Experiment Loop ---
# Parameter lists: number of interior control points and coverage (obstruction)
control_points_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # e.g. 1 means one interior point (2 segments), etc.
coverage_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # fraction of domain area covered by obstacles

num_trials = 3  # number of Monte Carlo trials per combination
results = []

for cp in control_points_list:
    for cov in coverage_list:
        trial_lengths = []
        for trial in range(num_trials):
            obstacles = generate_obstacles_coverage(cov, domain_size=domain_size, margin=0.1)
            x0 = initial_guess_perimeter(cp, domain_size)
            cons = {
                'type': 'ineq',
                'fun': lambda vars, obstacles=obstacles, pipe_radius=pipe_radius: constraint_wrapper(vars, obstacles,
                                                                                                     pipe_radius)
            }
            res = minimize(objective, x0, method='SLSQP', constraints=cons,
                           bounds=[(0, domain_size)] * (cp * 2),
                           options={'disp': False, 'maxiter': 500})
            trial_lengths.append(res.fun)
        avg_length = np.mean(trial_lengths)
        results.append((cp, cov, avg_length))
        print(f"Interior Points: {cp}, Coverage: {cov}, Avg. Optimized Length: {avg_length}")

results = np.array(results)  # shape: (N, 3) with columns [cp, coverage, avg_length]

# --- Plotting ---
plt.figure(figsize=(8, 6))
sc = plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis', s=100)
plt.colorbar(sc, label='Optimized Path Length')
plt.xlabel('Number of Interior Control Points')
plt.ylabel('Coverage (fraction of area obstructed)')
plt.title('Optimized Path Length vs. Control Points and Obstruction')

plt.legend()
plt.show(block=True)

# # Create a grid over the parameter space.
#
# cp_values = results[:, 0]
# cov_values = results[:, 1]
# length_values = results[:, 2]
#
# # Define a grid for contouring.
# xi = np.linspace(cp_values.min(), cp_values.max(), 100)
# yi = np.linspace(cov_values.min(), cov_values.max(), 100)
# XI, YI = np.meshgrid(xi, yi)

# Interpolate the optimized length values onto the grid.
ZI = griddata((cp_values, cov_values), length_values, (XI, YI), method='cubic')

plt.figure(figsize=(8, 6))
# Create filled contour plot.
contour_filled = plt.contourf(XI, YI, ZI, levels=50, cmap='viridis', vmin=1)
plt.colorbar(contour_filled, label='Optimized Path Length')
plt.xlabel('Number of Interior Control Points')
plt.ylabel('Coverage (Fraction of Area Obstructed)')
plt.title('Optimized Path Length Contour Plot')


plt.show(block=True)
