import numpy as np
import keyboard
import json
from scipy.spatial import Voronoi

GRID_SIZE = (200, 300)


def load_config(file_path):
    """Load configuration from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def set_grid_size(grid_size):
    """
    Sets the grid size for the simulation.
    grid_size: Tuple representing the dimensions of the grid.
    """
    global GRID_SIZE
    GRID_SIZE = grid_size


def stop_loop(event):
    global running
    running = False


# Helper functions
def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def reflective_boundaries(position, velocity, exclusion_center, exclusion_radius):
    """
    Reflects position and inverts velocity if an agent hits grid boundaries or enters an exclusion zone.
    Uses integer operations where applicable for better performance.
    """
    height, width = GRID_SIZE

    # Reflect against rectangular boundaries (with integer clamping)
    position[0] = min(max(position[0], 0), height)
    position[1] = min(max(position[1], 0), width)

    if position[0] == 0 or position[0] == height:
        velocity[0] = -velocity[0]
    if position[1] == 0 or position[1] == width:
        velocity[1] = -velocity[1]

    # Skip exclusion zone logic for performance unless explicitly needed
    if exclusion_center is not None and exclusion_radius is not None:
        dist_to_exclusion_center = np.linalg.norm(position - exclusion_center)
        if dist_to_exclusion_center < exclusion_radius:
            normal = (position - exclusion_center) / (dist_to_exclusion_center + 1e-5)
            velocity -= 2 * np.dot(velocity, normal) * normal
            position = exclusion_center + normal * exclusion_radius

    return position, velocity


def on_press(key):
    try:
        if key == keyboard.Key.esc:
            print("Escape pressed! Exiting loop.")
            return False  # Stop the listener
    except Exception as e:
        print(f"Error: {e}")


def area_polygon(vertices):
    """
    Function to calculate the area of a Voronoi region given its vertices.

    Parameters
    ==========
    vertices : Coordinates (array, 2 dimensional).
    """

    N, dim = vertices.shape

    # dim is 2.
    # Vertices are listed consecutively.

    A = 0

    for i in range(N - 1):
        # Below is the formula of the area of a triangle given the vertices.
        A += np.abs(
            vertices[-1, 0] * (vertices[i, 1] - vertices[i + 1, 1])
            + vertices[i, 0] * (vertices[i + 1, 1] - vertices[-1, 1])
            + vertices[i + 1, 0] * (vertices[-1, 1] - vertices[i, 1])
        )

    A *= 0.5

    return A


def global_clustering(x, y, Rf, L):
    """
    Function to calculate the global alignment coefficient.

    Parameters
    ==========
    x, y : Positions.
    Rf : Flocking radius.
    L : Dimension of the squared arena.
    """

    N = np.size(x)

    # Use the replicas of all points to calculate Voronoi for
    # a more precise estimate.
    points = np.zeros([9 * N, 2])

    for i in range(3):
        for j in range(3):
            s = 3 * i + j
            points[s * N : (s + 1) * N, 0] = x + (j - 1) * L
            points[s * N : (s + 1) * N, 1] = y + (i - 1) * L

    # The format of points is the one needed by Voronoi.
    # points[:, 0] contains the x coordinates
    # points[:, 1] contains the y coordinates

    vor = Voronoi(points)
    """
    vertices = vor.vertices  # Voronoi vertices.
    regions = vor.regions  # Region list. 
    # regions[i]: list of the vertices indices for region i.
    # If -1 is listed: the region is open (includes point at infinity).
    point_region = vor.point_region  # Region associated to input point.
    """

    # Consider only regions of original set of points (no replicas).
    list_regions = vor.point_region[4 * N : 5 * N]

    c = 0

    for i in list_regions:
        indices = vor.regions[i]
        # print(f'indices = {indices}')
        if len(indices) > 0:
            if np.size(np.where(np.array(indices) == -1)[0]) == 0:
                # Region is finite.
                # Calculate area.
                A = area_polygon(vor.vertices[indices, :])
                if A < np.pi * Rf**2:
                    c += 1

    c = c / N

    return c
