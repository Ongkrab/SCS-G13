import numpy as np
import keyboard
import json

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
