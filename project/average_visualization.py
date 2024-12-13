import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
FOLDER_NAMES = [
    "20241210-182438",
    "20241210-182514",
    "20241210-182629",
    "20241210-183523",
    "20241210-183554",
]
IMAGE_FOLDER_NAME = "images"


def convert_to_title_case(s):
    return s.replace("_", " ").title()


def read_results(folder_name, root_path, group_by="intrusion.radius"):
    reindeer_population_results = []
    predator_population_results = []
    death_by_age_results = []
    death_by_culling_results = []
    death_by_predator_results = []
    death_by_starvation_results = []
    predator_death_by_age_results = []
    predator_death_by_starvation_results = []
    culling_statistics_results = []

    intrusion_radius_configs = []
    food_regeneration_rate_configs = []

    for folder_name in FOLDER_NAMES:
        config = helper.load_config(f"{root_path}{folder_name}/config.json")

        ## Update this when you change the group_by parameter
        intrusion_raidus = config["intrusion"]["radius"]
        food_regeneration_rate = config["simulation"]["food_regeneration_rate"]

        intrusion_radius_configs.append(intrusion_raidus)
        food_regeneration_rate_configs.append(food_regeneration_rate)

        reindeer_population = genfromtxt(
            f"{root_path}{folder_name}/reindeer_population.csv", delimiter=","
        )
        predator_population = genfromtxt(
            f"{root_path}{folder_name}/predator_population.csv", delimiter=","
        )
        death_by_age = genfromtxt(
            f"{root_path}{folder_name}/death_by_age.csv", delimiter=","
        )
        death_by_culling = genfromtxt(
            f"{root_path}{folder_name}/death_by_culling.csv", delimiter=","
        )
        death_by_predator = genfromtxt(
            f"{root_path}{folder_name}/death_by_predator.csv", delimiter=","
        )
        death_by_starvation = genfromtxt(
            f"{root_path}{folder_name}/death_by_starvation.csv", delimiter=","
        )
        predator_death_by_age = genfromtxt(
            f"{root_path}{folder_name}/predator_death_by_age.csv", delimiter=","
        )
        predator_death_by_starvation = genfromtxt(
            f"{root_path}{folder_name}/predator_death_by_starvation.csv", delimiter=","
        )
        culling_statistics = genfromtxt(
            f"{root_path}{folder_name}/culling_statistics.csv", delimiter=","
        )

        reindeer_population_results.append(reindeer_population)
        predator_population_results.append(predator_population)
        death_by_age_results.append(death_by_age)
        death_by_culling_results.append(death_by_culling)
        death_by_predator_results.append(death_by_predator)
        death_by_starvation_results.append(death_by_starvation)
        predator_death_by_age_results.append(predator_death_by_age)
        predator_death_by_starvation_results.append(predator_death_by_starvation)
        culling_statistics_results.append(culling_statistics)

    data = {
        "reindeer_population": reindeer_population_results,
        "predator_population": predator_population_results,
        "death_by_age": death_by_age_results,
        "death_by_culling": death_by_culling_results,
        "death_by_predator": death_by_predator_results,
        "death_by_starvation": death_by_starvation_results,
        "predator_death_by_age": predator_death_by_age_results,
        "predator_death_by_starvation": predator_death_by_starvation_results,
        "culling_statistics": culling_statistics_results,
        "intrusion_radius": intrusion_radius_configs,
        "food_regeneration_rate": food_regeneration_rate_configs,
    }

    df = pd.DataFrame(data)

    return df


def average_population_dynamics(
    df, group_by="intrusion_radius", is_save=False, image_folder_path=IMAGE_FOLDER_NAME
):
    # Group the data by the specified column
    grouped_data = df.groupby(group_by).agg(
        {
            "reindeer_population": "mean",
            "predator_population": "mean",
        }
    )

    average_reindeer_population = grouped_data["reindeer_population"]
    average_predator_population = grouped_data["predator_population"]

    label = convert_to_title_case(group_by)
    save_file_name = f"{image_folder_path}/average_population_dynamics_{group_by}.svg"

    # Plot the population dynamics
    plt.figure(figsize=(12, 8))
    for intrusion_radius in average_reindeer_population.index:
        plt.plot(
            average_reindeer_population[intrusion_radius],
            label=f"Reindeer Population - {label}: {intrusion_radius}",
        )
        plt.plot(
            average_predator_population[intrusion_radius],
            label=f"Predator Population - {label}: {intrusion_radius}",
        )

    max_steps = len(average_reindeer_population[intrusion_radius])
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
    if is_save:
        plt.savefig(save_file_name)
    plt.show()


def average_culling_statistics(
    df,
    group_by="intrusion_radius",
    max_steps=10000,
    is_save=False,
    image_folder_path="",
):

    label = convert_to_title_case(group_by)
    save_file_name = f"{image_folder_path}/average_culling_statistics.svg"

    # Plot the culling statistics
    plt.figure(figsize=(12, 8))

    for culling_statistics in df["culling_statistics"]:
        print(culling_statistics)
        intrusion_radius = culling_statistics[0][0]
        plt.plot(
            culling_statistics[:, 0],
            culling_statistics[:, 1],
            label=f"Culling Statistics - {label}: {intrusion_radius}",
        )
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
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if is_save:
        plt.savefig(save_file_name)
    plt.show()


if __name__ == "__main__":
    df = read_results(FOLDER_NAMES, ROOT_PATH, group_by="intrusion.radius")
    average_population_dynamics(df, group_by="intrusion_radius")
    average_population_dynamics(df, group_by="food_regeneration_rate")
    # average_culling_statistics(df, group_by="intrusion_radius") # Haven't Average Culling Statistics yet
