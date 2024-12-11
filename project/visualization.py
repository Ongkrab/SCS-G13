import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from helper import *
import pandas as pd

ROOT_PATH = "./results/"

FOLDER_NAME_DEFAULT = "20241209-145755"
IMAGE_FOLDER_NAME = "images"


def create_population_dynamic_plot(
    reindeer_population,
    predator_population,
    predator_reintroduction,
    latest_step,
    max_steps,
    is_save=False,
    image_folder_path="",
):

    plt.figure()
    plt.plot(reindeer_population, label="Reindeer Population", color="blue")
    plt.plot(predator_population, label="Predator Population", color="red")
    if latest_step > max_steps / 2:
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
    culling_statistics,
    latest_step,
    max_steps,
    is_save=False,
    image_folder_path="",
):
    culling_statistics = np.array(culling_statistics)
    plt.plot(culling_statistics[:, 0], culling_statistics[:, 1])
    if latest_step > max_steps / 2:
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


def create_death_plot(
    death_by_age,
    death_by_starvation,
    death_by_predator,
    death_by_culling,
    latest_step,
    max_steps,
    is_save=False,
    image_folder_path="",
):
    plt.plot(
        death_by_age[:, 0],
        death_by_age[:, 1],
        color="blue",
        label="Old age",
    )
    plt.plot(
        death_by_starvation[:, 0],
        death_by_starvation[:, 1],
        color="green",
        label="Starved",
    )
    plt.plot(
        death_by_predator[:, 0], death_by_predator[:, 1], color="red", label="Eaten"
    )

    plt.plot(
        death_by_culling[:, 0], death_by_culling[:, 1], color="black", label="Culled"
    )
    if latest_step > max_steps / 2:
        plt.axvline(
            x=max_steps / 2,
            color="grey",
            linestyle="--",
            linewidth=2,
            label="Intrusion added",
        )
    plt.title("Prey cause of death")
    plt.xlabel("Time Step")
    plt.ylabel("Total amount")
    plt.legend()
    if is_save:
        plt.savefig(image_folder_path + "death_plot.png")
    plt.show()

    # Define the interval for comparison
    interval = 50
    # Interpolate data for death_by_age

    temp_death_by_age = []
    for i in range(len(death_by_age) - 1):
        for j in range(int(death_by_age[i][0]), int(death_by_age[i + 1][0])):
            temp_death_by_age.append(
                [
                    j,
                    death_by_age[i][1]
                    + (j - death_by_age[i][0])
                    * (death_by_age[i + 1][1] - death_by_age[i][1])
                    / (death_by_age[i + 1][0] - death_by_age[i][0]),
                ]
            )
    temp_death_by_age = np.array(temp_death_by_age)
    df = pd.DataFrame({"x": temp_death_by_age[:, 0], "y": temp_death_by_age[:, 1]})
    # Compute the average change over the interval
    df["y_change"] = df["y"].diff(periods=interval) / interval
    plt.plot(df["x"], df["y_change"], label="Old age", color="blue")

    # Interpolate data for death_by_culling
    temp_death_by_culling = []
    for i in range(len(death_by_culling) - 1):
        for j in range(int(death_by_culling[i][0]), int(death_by_culling[i + 1][0])):
            temp_death_by_culling.append(
                [
                    j,
                    death_by_culling[i][1]
                    + (j - death_by_culling[i][0])
                    * (death_by_culling[i + 1][1] - death_by_culling[i][1])
                    / (death_by_culling[i + 1][0] - death_by_culling[i][0]),
                ]
            )
    temp_death_by_culling = np.array(temp_death_by_culling)
    df = pd.DataFrame(
        {"x": temp_death_by_culling[:, 0], "y": temp_death_by_culling[:, 1]}
    )
    # Compute the average change over the interval
    df["y_change"] = df["y"].diff(periods=interval) / interval
    plt.plot(df["x"], df["y_change"], label="Culled", color="orange")

    # Starvation
    df = pd.DataFrame({"x": death_by_starvation[:, 0], "y": death_by_starvation[:, 1]})
    # Compute the average change over the interval
    df["y_change"] = df["y"].diff(periods=interval) / interval
    plt.plot(df["x"], df["y_change"], label="Starved", color="green")

    # Predator
    df = pd.DataFrame({"x": death_by_predator[:, 0], "y": death_by_predator[:, 1]})
    # Compute the average change over the interval
    df["y_change"] = df["y"].diff(periods=interval) / interval
    plt.plot(df["x"], df["y_change"], label="Eaten", color="red")

    if latest_step > max_steps / 2:
        plt.axvline(
            x=max_steps / 2,
            color="grey",
            linestyle="--",
            linewidth=2,
            label="Intrusion added",
        )
    plt.title(
        f"Rolling average cause of death per timestep over the last {interval} timesteps"
    )
    plt.xlabel("Time Step")
    plt.legend()
    if is_save:
        plt.savefig(image_folder_path + "death_plot_average.png")
    plt.show()


def create_predator_death_by_age(
    predator_death_by_age,
    predator_death_by_starvation,
    latest_step,
    max_steps,
    is_save=False,
    image_folder_path="",
):
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
    if latest_step > max_steps / 2:

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
        plt.savefig(image_folder_path + "predator_death_by_age.png")

    plt.show()


def reindeer_clustering_coefficient_plot(
    reindeer_clustering_coefficient,
    reindeer_population,
    predator_population,
    latest_step,
    max_steps,
    is_save=False,
    image_folder_path="",
):
    if len(reindeer_clustering_coefficient) == 0:
        print("No clustering coefficient data")
        return
    cluster_color = "tab:brown"
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Clustering coefficient")
    ax1.plot(
        reindeer_clustering_coefficient,
        label="Reindeer Clustering Coefficient",
        color=cluster_color,
    )
    ax1.tick_params(axis="y", labelcolor=cluster_color)
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(reindeer_population, label="Reindeer Population", color="blue")
    ax2.plot(
        predator_population,
        label="Predator Population",
        color="orange",
    )
    ax2.set_ylabel("Number of Population")

    if latest_step > max_steps / 2:
        plt.axvline(
            x=max_steps / 2,
            color="grey",
            linestyle="--",
            linewidth=2,
            label="Intrusion added",
        )

    fig.tight_layout()

    plt.title("Reindeer clustering coefficient")
    plt.legend()
    if is_save:
        plt.savefig(image_folder_path + "reindeer_clustering_coefficient.png")
    plt.show()


def visualize(root_path=ROOT_PATH, folder_name=FOLDER_NAME_DEFAULT):
    result_folder_path = root_path + folder_name + "/"
    config_path = result_folder_path + "config.json"
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

    reindeer_clustering_coeefficient = genfromtxt(
        result_folder_path + "reindeer_clustering_coefficient.csv", delimiter=","
    )

    predator_reintroduction = predator_reintroduction.reshape(-1, 2)

    latest_step = len(reindeer_population)
    config = load_config(config_path)
    simulation = config["simulation"]
    max_steps = simulation["max_steps"]
    predator_reintroduction = predator_reintroduction.reshape(-1, 2)

    create_population_dynamic_plot(
        reindeer_population,
        predator_population,
        predator_reintroduction,
        latest_step,
        max_steps,
        True,
        image_folder_path,
    )

    create_death_plot(
        death_by_age,
        death_by_starvation,
        death_by_predator,
        death_by_culling,
        latest_step,
        max_steps,
        True,
        image_folder_path,
    )

    create_predator_death_by_age(
        death_by_age,
        death_by_starvation,
        latest_step,
        max_steps,
        True,
        image_folder_path,
    )
    create_culling_statistics_plot(
        culling_statistics, latest_step, max_steps, True, image_folder_path
    )

    reindeer_clustering_coefficient_plot(
        reindeer_clustering_coeefficient,
        reindeer_population,
        predator_population,
        latest_step,
        max_steps,
        True,
        image_folder_path,
    )


if __name__ == "__main__":
    visualize()
