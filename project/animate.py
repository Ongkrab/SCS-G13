import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


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
        plt.figure(figsize=(8, 4))
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
                label="Preys",
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
        plt.tight_layout()
        if isAnimate:
            plt.pause(0.00001)
        if isCapture:
            plt.savefig(result_image_path + f"step_{step}.svg")
            plt.close()
        plt.clf()


def plot_record_step(
    grid_size,
    intrusion_center,
    intrusion_radius,
    food_grid,
    reindeers,
    predators,
    step,
    ax,
    isAnimate=True,
    isRecord=False,
    capture_interval=10000,
):
    isCapture = step % capture_interval == 0
    if isAnimate or isCapture:
        ax.clear()
        ax.imshow(
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
            ax.add_patch(circle)

        # Plot reindeers with color representing their age
        ages = [reindeer.age for reindeer in reindeers]
        max_age = max(ages) if ages else 1
        colors = plt.cm.Blues(
            np.array(ages) / max_age
        )  # Normalize ages to [0, 1] and map to Blues colors
        alphas = 0.3 + 0.6 * (
            1 - np.array(ages) / max_age
        )  # Normalize ages to [0, 1] and map to alpha range [0.3, 0.9]

        for reindeer, color, alpha in zip(reindeers, colors, alphas):
            ax.scatter(
                reindeer.position[1],
                reindeer.position[0],
                color=color,
                alpha=alpha,
                edgecolor="black",
            )

        # Plot predators
        for predator in predators:
            ax.scatter(
                predator.position[1],
                predator.position[0],
                color="red",
                edgecolor="black",
            )

        ax.set_title(f"Step: {step}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")


def save_simulation_video(
    grid_size,
    intrusion_center,
    intrusion_radius,
    food_grid,
    result_video_path,
    reindeers_list,
    predators_list,
    steps,
    capture_interval=10000,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    def update(frame):
        step = steps[frame]
        food_grid_frame = food_grid[frame]
        reindeers = reindeers_list[frame]
        predators = predators_list[frame]

        plot_record_step(
            grid_size,
            intrusion_center,
            intrusion_radius,
            food_grid_frame,
            reindeers,
            predators,
            step,
            ax,
            isAnimate=True,
            isRecord=False,
            capture_interval=capture_interval,
        )

    anim = FuncAnimation(fig, update, frames=len(steps), repeat=False)
    writer = FFMpegWriter(fps=10)
    anim.save(result_video_path, writer=writer)
