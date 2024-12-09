import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

ROOT_PATH = "./hello2/"
FOLDER_NAME_DEFAULT = "20241206-133229"
IMAGE_FOLDER_NAME = "images"


def create_population_dynamic_plot(
    reindeer_population,
    predator_population,
    predator_reintroduction,
    is_save=False,
    image_folder_path="",
):
    max_steps = len(reindeer_population)
    plt.figure()
    plt.plot(reindeer_population, label="Reindeer Population", color="blue")
    plt.plot(predator_population, label="Predator Population", color="red")
    # if step > max_steps / 2:
    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )
    if len(predator_reintroduction) > 0:
        plt.scatter(
            predator_reintroduction[:, 0],
            predator_reintroduction[:, 1],
            c="black",
            label="Predator reintroduced",
            alpha=1,
        )
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Population Dynamics")
    plt.legend()
    if is_save:
        plt.savefig(image_folder_path + "population_dynamics.png")
    plt.show()


def create_culling_statistics_plot(
    culling_statistics, max_steps, is_save=False, image_folder_path=""
):
    # max_steps = len(culling_statistics)
    culling_statistics = np.array(culling_statistics)
    plt.plot(culling_statistics[:, 0], culling_statistics[:, 1])
    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )
    plt.title("Culling statistics")
    plt.xlabel("Time Step")
    plt.ylabel("Amount culled each season")
    if is_save:
        plt.savefig(image_folder_path + "culling_statistics.png")
    plt.show()


def create_predator_death_plot(
    predator_death_by_age,
    predator_death_by_starvation,
    is_save=False,
    image_folder_path="",
):
    max_steps = len(predator_death_by_starvation)
    plt.plot(
        predator_death_by_age[:, 0],
        predator_death_by_age[:, 1],
        color="blue",
        label="Old age",
    )
    plt.plot(
        predator_death_by_starvation[:, 0],
        predator_death_by_starvation[:, 1],
        color="green",
        label="Starved",
    )
    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )
    plt.title("Predator cause of death")
    plt.xlabel("Time Step")
    plt.ylabel("Total amount")
    plt.legend()
    if is_save:
        plt.savefig(image_folder_path + "predator_death.png")
    plt.show()


def visualize(root_path=ROOT_PATH, folder_name=FOLDER_NAME_DEFAULT):
    result_folder_path = root_path + folder_name + "/"

    image_folder_path = result_folder_path + IMAGE_FOLDER_NAME + "/"

    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
    print("Visualizing results from folder: " + result_folder_path)
    reindeer_population = genfromtxt(
        result_folder_path + "reindeer_population.csv", delimiter=","
    )
    predator_population = genfromtxt(
        result_folder_path + "predator_population.csv", delimiter=","
    )
    predator_reintroduction = genfromtxt(
        result_folder_path + "predator_reintroduction.csv", delimiter=","
    )
    death_by_age = genfromtxt(result_folder_path + "death_by_age.csv", delimiter=",")
    death_by_starvation = genfromtxt(
        result_folder_path + "death_by_starvation.csv", delimiter=","
    )
    death_by_predator = genfromtxt(
        result_folder_path + "death_by_predator.csv", delimiter=","
    )
    death_by_culling = genfromtxt(
        result_folder_path + "death_by_culling.csv", delimiter=","
    )
    culling_statistics = genfromtxt(
        result_folder_path + "culling_statistics.csv", delimiter=","
    )
    max_steps = len(reindeer_population)
    create_population_dynamic_plot(
        reindeer_population,
        predator_population,
        predator_reintroduction,
        True,
        image_folder_path,
    )
    create_predator_death_plot(
        death_by_age, death_by_starvation, True, image_folder_path
    )
    create_culling_statistics_plot(
        culling_statistics, max_steps, True, image_folder_path
    )
