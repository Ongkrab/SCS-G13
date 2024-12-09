import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
FOLDER_NAMES = ["20241209-133734", "20241209-132448"]
IMAGE_FOLDER_NAME = "images"

def create_population_dynamic_plot_multi_run(
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
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()



if __name__ == "__main__":
    create_population_dynamic_plot_multi_run(FOLDER_NAMES, ROOT_PATH)