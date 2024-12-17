import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/Ong_results/results/"
CONFIG_PATH = "./config.json"
FOLDER_NAMES = ["seed5_intrusion0", "seed5_intrusion60"]
IMAGE_FOLDER_NAME = "images"

def create_population_dynamics_multi_run(
    FOLDER_NAMES,
    ROOT_PATH,
    max_steps=10000,
    is_save=False,
    image_folder_path="",
):
    
    """
    Visualize population dynamics for multiple runs on the same graph.

    :param multi_run_data: A list of dictionaries, each containing data for a single run.
                           Each dictionary should have 'reindeer_population', 'predator_population',
                           and 'intrusion_radius' keys.
    :param max_steps: Maximum number of steps for the simulation.
    :param is_save: Whether to save the plot as an image.
    :param image_folder_path: Path to save the image.
    """

    plt.figure(figsize=(12, 8))

    for i, folder_name in enumerate(FOLDER_NAMES):

        reindeer_population = genfromtxt(f"{ROOT_PATH}{folder_name}/reindeer_population.csv", delimiter=',')
        predator_population = genfromtxt(f"{ROOT_PATH}{folder_name}/predator_population.csv", delimiter=',')
        config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
        intrusion_radius = config["intrusion"]["radius"]

        plt.plot(reindeer_population, label=f"Reindeer Population - Intrusion Radius: {intrusion_radius}")
        plt.plot(predator_population, label=f"Predator Population - Intrusion Radius: {intrusion_radius}")
        

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Population Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_culling_statistics_multi_run(
        FOLDER_NAMES,
        ROOT_PATH,
        max_steps=10000,
        is_save=False,
        image_folder_path="",
):
    plt.figure(figsize=(12, 8))
    for i, folder_name in enumerate(FOLDER_NAMES):
        culling_statistics = genfromtxt(f"{ROOT_PATH}{folder_name}/culling_statistics.csv", delimiter=',')
        config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
        intrusion_radius = config["intrusion"]["radius"]
        plt.plot(culling_statistics[:, 0], culling_statistics[:, 1], label=f"Culling Statistics - Intrusion Radius: {intrusion_radius}")
    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )
    plt.xlabel("Time Step")
    plt.ylabel("Culling Statistics")
    plt.title("Culling Statistics for Different Intrusion Radii")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_population_dynamics_multi_run(FOLDER_NAMES, ROOT_PATH)
    create_culling_statistics_multi_run(FOLDER_NAMES, ROOT_PATH)