import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
INTRUSION_INTEREST = [10, 50, 60, 70, 80]
FOLDER_NAMES = [
    "20241213-131030",
    "20241213-131101",
    "20241213-131132",
    "20241213-131226",
    "20241213-131258",
    "20241213-131325",
    "20241213-131337",
    "20241213-131437",
    "20241213-131920",
    "20241213-131928",
    "20241213-132148",
    "20241213-132224",
    "20241213-132327",
    "20241213-132348",
    "20241213-132521",
    "20241213-132757",
    "20241213-132928",
    "20241213-133002",
    "20241213-133042",
    "20241213-133127",
    "20241213-133243",
    "20241213-133713",
    "20241213-133747",
    "20241213-133836",
    "20241213-133918",
    "20241213-134016",
    "20241213-134115",
    "20241213-134225",
    "20241213-134339",
    "20241213-134500",
    "20241213-134632",
    "20241213-134733",
    "20241213-134838",
    "20241213-135031",
    "20241213-135110",
    "20241213-135657",
    "20241213-135717",
    "20241213-135728",
    "20241213-140110",
    "20241213-140221",
    "20241213-141231",
    "20241213-141335",
    "20241213-141424",
    "20241213-141429",
    "20241213-141620",
    "20241213-141638",
    "20241213-141649",
    "20241213-141656",
    "20241213-142033",
    "20241213-142103",
    "20241213-142558",
    "20241213-142705",
    "20241213-142809",
    "20241213-142829",
    "20241213-143009",
    "20241213-143050",
    "20241213-143109",
    "20241213-143128",
    "20241213-143152",
    "20241213-143158",
    "20241213-143936",
    "20241213-143942",
    "20241213-144032",
    "20241213-144044",
    "20241213-144157",
    "20241213-144304",
    "20241213-144352",
    "20241213-144406",
    "20241213-144544",
    "20241213-144635",
    "20241213-145300",
    "20241213-145323",
    "20241213-145342",
    "20241213-145408",
    "20241213-145447",
    "20241213-145534",
    "20241213-145620",
    "20241213-145710",
    "20241213-150007",
    "20241213-150011",
    "20241213-150248",
    "20241213-150313",
    "20241213-150403",
    "20241213-150414",
    "20241213-150542",
    "20241213-150642",
    "20241213-150703",
    "20241213-150841",
    "20241213-150935",
    "20241213-150942",
]
IMAGE_FOLDER_NAME = "merges/images"


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
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/average_population_dynamics_{group_by}.svg"
    )

    # Plot the population dynamics
    plt.figure(figsize=(12, 8))
    for intrusion_radius in average_reindeer_population.index:
        if intrusion_radius not in INTRUSION_INTEREST:
            continue
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


import matplotlib.pyplot as plt


def error_bar_population_dynamics(
    df, group_by="intrusion_radius", is_save=False, image_folder_path=IMAGE_FOLDER_NAME
):
    # Group the data by the specified column and calculate the mean and standard deviation
    grouped_data = df.groupby(group_by).agg(
        {
            "reindeer_population": ["mean"],
            "predator_population": ["mean"],
        }
    )

    mean_reindeer_population = grouped_data["reindeer_population"]["mean"]
    mean_predator_population = grouped_data["predator_population"]["mean"]

    mean_reindeers = []
    mean_predators = []
    std_reindeers = []
    std_predators = []

    for intrusion_radius in mean_reindeer_population.index:
        mean_reindeer = mean_reindeer_population[intrusion_radius].mean()
        std_reindeer = mean_reindeer_population[intrusion_radius].std()

        mean_predator = mean_predator_population[intrusion_radius].mean()
        std_predator = mean_predator_population[intrusion_radius].std()

        mean_reindeers.append(mean_reindeer)
        mean_predators.append(mean_predator)
        std_reindeers.append(std_reindeer)
        std_predators.append(std_predator)

    label = convert_to_title_case(group_by)
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/error_bar_population_dynamics_{group_by}.svg"
    )

    x_axis = mean_reindeer_population.index
    # Plot the mean population dynamics with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x_axis,
        mean_reindeers,
        yerr=std_reindeers,
        label="Reindeer Population",
        fmt="o",
        capsize=5,
    )
    plt.errorbar(
        x_axis,
        mean_predators,
        yerr=std_predators,
        label="Predator Population",
        fmt="o",
        capsize=5,
    )
    plt.xlabel(label)
    plt.ylabel("Population")
    plt.title(f"Population Dynamics with Error Bars by {label}")
    plt.legend()
    plt.grid(True)

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
    # Merge the culling statistics with the configuration data to get the intrusion radius
    # Group the data by the specified column and calculate the mean of the second column

    grouped_data = df.groupby(group_by).agg(
        {
            "culling_statistics": ["mean"],
        }
    )

    average_culling_statistic = grouped_data["culling_statistics"]["mean"]

    label = convert_to_title_case(group_by)
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/average_culling_statistics_{group_by}.svg"
    )

    plt.figure(figsize=(12, 8))

    for intrusion_radius in average_culling_statistic.index:
        if intrusion_radius not in INTRUSION_INTEREST:
            continue
        x_value = average_culling_statistic[intrusion_radius][:, 0]
        y_value = average_culling_statistic[intrusion_radius][:, 1]

        plt.plot(  # Plot the average culling statistics
            x_value,
            y_value,
            label=f"Culling Statistic - {label}: {intrusion_radius}",
        )
    plt.xlabel(label)
    plt.ylabel("Average Culling Statistic")
    plt.title(f"Average Culling Statistic by {label}")
    plt.legend()
    plt.grid(True)

    if is_save:
        plt.savefig(save_file_name)
    plt.show()


if __name__ == "__main__":
    df = read_results(FOLDER_NAMES, ROOT_PATH, group_by="intrusion.radius")
    average_population_dynamics(df, group_by="intrusion_radius", is_save=True)
    error_bar_population_dynamics(df, group_by="intrusion_radius", is_save=True)
    average_culling_statistics(
        df, group_by="intrusion_radius", is_save=True
    )  # Haven't Average Culling Statistics yet
