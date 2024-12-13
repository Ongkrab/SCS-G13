import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"

FOLDER_NAMES_LIST = [["1","2","3","4","5","6","7","8","9","10"], \\
                     ["1i","2i","3i","4i","5i","6i","7i","8i","9i","10i"], \\
                     ["1i","2i","3i","4i","5i","6i","7i","8i","9i","10i"], \\
                     ["1i","2i","3i","4i","5i","6i","7i","8i","9i","10i"]]

IMAGE_FOLDER_NAME = "images"

def create_population_dynamics_multi_run(
    FOLDER_NAMES_LIST,
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

    plt.figure(figsize=(10, 5))

    for folder_names_current in FOLDER_NAMES_LIST:
        reindeer_population=None
        predator_population=None
        for i, folder_name in enumerate(folder_names_current):
            reindeer_population_temp = genfromtxt(f"{ROOT_PATH}{folder_name}/reindeer_population.csv", delimiter=',')
            if reindeer_population is None:
                reindeer_population = np.zeros_like(reindeer_population_temp)
            reindeer_population+=reindeer_population_temp
            predator_population_temp = genfromtxt(f"{ROOT_PATH}{folder_name}/predator_population.csv", delimiter=',')
            if predator_population is None:
                predator_population = np.zeros_like(predator_population_temp)
            predator_population+=predator_population_temp
            
            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
            intrusion_radius = config["intrusion"]["radius"]
        reindeer_population/=len(folder_names_current)
        predator_population/=len(folder_names_current)
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
    plt.title(f"Average Population Dynamics over {len(FOLDER_NAMES_LIST[0])} simulation pairs")
    plt.legend()
    plt.tight_layout()#rect=[1, 1, 0, 0])
    plt.show()

def create_culling_statistics_multi_run(
        FOLDER_NAMES_LIST,
        ROOT_PATH,
        max_steps=10000,
        is_save=False,
        image_folder_path="",
):
    plt.figure(figsize=(10, 5))

    for folder_names_current in FOLDER_NAMES_LIST:
        culling_statistics = None
        for i, folder_name in enumerate(folder_names_current):
            culling_statistics_temp = genfromtxt(f"{ROOT_PATH}{folder_name}/culling_statistics.csv", delimiter=',')
            if culling_statistics is None:
                culling_statistics = np.zeros_like(culling_statistics_temp)
            culling_statistics+=culling_statistics_temp
            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
            intrusion_radius = config["intrusion"]["radius"]
        culling_statistics/=len(folder_names_current)
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
    plt.title(f"Average Culling Statistics over {len(FOLDER_NAMES_LIST[0])} simulation pairs")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    dif_list = []
    for k in range(len(FOLDER_NAMES_LIST)):
        dif=0
        for i, folder_name in enumerate(FOLDER_NAMES_LIST[k]):
            death_by_culling = genfromtxt(f"{ROOT_PATH}{folder_name}/death_by_culling.csv", delimiter=',')
            dif+=death_by_culling[250][1]-death_by_culling[126][1] 
        dif/=len(FOLDER_NAMES_LIST[k])
        dif_list.append(dif)

    print(dif_list)
    
if __name__ == "__main__":
    create_population_dynamics_multi_run(FOLDER_NAMES_LIST, ROOT_PATH)
    create_culling_statistics_multi_run(FOLDER_NAMES_LIST, ROOT_PATH)
