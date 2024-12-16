import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_simulation_step(
    grid_size,
    intrusion_center,
    intrusion_radius,
    food_grid,
    result_image_path,
    reindeers,
    predators,
    step,
    isAnimate=True,
    capture_interval=10000,
):
    isCapture = step % capture_interval == 0
    if isAnimate == True or isCapture == True:
        plt.figure(figsize=(10, 5))
        plt.imshow(
            food_grid,
            cmap="Greens",
            extent=(0, grid_size[1], 0, grid_size[0]),
        )
        if intrusion_center is not None and intrusion_radius is not None:
            circle = Circle(
                (intrusion_center[1], intrusion_center[0]),
                intrusion_radius,
                color="grey",
                alpha=1,
            )
            plt.gca().add_artist(circle)
        if reindeers:
            reindeer_positions = np.array([r.position for r in reindeers])
            reindeer_alphas = np.array([r.get_alpha() for r in reindeers])
            plt.scatter(
                reindeer_positions[:, 1],
                reindeer_positions[:, 0],
                c="blue",
                label="Reindeer",
                alpha=reindeer_alphas[:],
            )
        if predators:
            predator_positions = np.array([p.position for p in predators])
            predator_alphas = np.array([p.get_alpha() for p in predators])
            plt.scatter(
                predator_positions[:, 1],
                predator_positions[:, 0],
                c="red",
                label="Predators",
                alpha=predator_alphas[:],
            )

        plt.title(f"Step {step}")
        plt.legend()
        if isAnimate:
            plt.pause(0.00001)
            # plt.pause(0.01)
        if isCapture:
            plt.savefig(result_image_path + f"step_{step}.svg")
            plt.close()
        plt.clf()
