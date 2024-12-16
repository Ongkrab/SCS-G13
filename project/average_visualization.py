import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
INTRUSION_INTEREST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# FOLDER_NAMES = [
#     "20241213-144635",
#     "20241213-144544",
#     "20241213-144406",
#     "20241213-144352",
#     "20241213-144304",
#     "20241213-144157",
#     "20241213-144044",
#     "20241213-144032",
#     "20241213-143942",
#     "20241213-143936",
#     "20241214-233843",
#     "20241214-233947",
#     "20241214-234001",
#     "20241214-234009",
#     "20241214-234039",
#     "20241214-234223",
#     "20241214-234326",
#     "20241214-234335",
#     "20241214-234524",
#     "20241214-234532",
# ]  # Intrusion Radius 80, Food regeneration 0.0025, 0.0035
# FOLDER_NAMES = [
#     "20241213-150007",
#     "20241213-150011",
#     "20241213-145710",
#     "20241213-145620",
#     "20241213-145534",
#     "20241213-145447",
#     "20241213-145408",
#     "20241213-145342",
#     "20241213-145323",
#     "20241213-145300",
#     "20241214-232758",
#     "20241214-232843",
#     "20241214-232921",
#     "20241214-232942",
#     "20241214-233017",
#     "20241214-233043",
#     "20241214-233058",
#     "20241214-233113",
#     "20241214-233131",
#     "20241214-233151",
# ]  # Intrusion Radius 70, Food regeneration 0.0025, 0.0035
# FOLDER_NAMES = [
#     "20241215-102450",
#     "20241215-102454",
# ]
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
    "20241214-224857",
    "20241214-225523",
    "20241214-225534",
    "20241214-225540",
    "20241214-225551",
    "20241214-225601",
    "20241214-225606",
    "20241214-225612",
    "20241214-225616",
    "20241214-225622",
]
# FOLDER_NAMES = [
#     "20241214-225534",
#     "20241213-142809",
# ]
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
    df,
    group_by="intrusion_radius",
    only_reindeer=False,
    is_save=False,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
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
        if intrusion_radius not in intrusion_interest:
            continue
        plt.plot(
            average_reindeer_population[intrusion_radius],
            label=f"Reindeer Population - {label}: {intrusion_radius}",
        )

        if not only_reindeer:
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
    plt.title("Population Dynamics Intrusion Radius")
    plt.legend()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if is_save:
        plt.savefig(save_file_name)
    plt.show()


import matplotlib.pyplot as plt


def average_population_dynamics_2axis(
    df,
    group_by="intrusion_radius",
    only_reindeer=False,
    is_save=False,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
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
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Reindeer Population", color="tab:blue")
    ax2.set_ylabel("Predator Population", color="tab:red")
    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
    ]
    i = 0
    for intrusion_radius in average_reindeer_population.index:
        if intrusion_radius not in intrusion_interest:
            continue
        ax1.plot(
            average_reindeer_population[intrusion_radius],
            label=f"Reindeer Population - {label}: {intrusion_radius}",
            color=colors[i],
        )

        if not only_reindeer:
            ax2.plot(
                average_predator_population[intrusion_radius],
                label=f"Predator Population - {label}: {intrusion_radius}",
                color=colors[i + 1],
            )
        i += 2
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_ylim(0, 350)
    ax2.set_ylim(0, 50)
    ax1.set_xlim(0, 10000)
    ax2.set_ylabel("Predator Population")

    max_steps = len(average_reindeer_population[intrusion_radius])
    ax1.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    plt.xlabel("Time Step")

    plt.title("Average Population Dynamics for Intrusion Radius: 60")
    # plt.title("Population Dynamics for Intrusion Radius: 60")
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    if is_save:
        plt.savefig(save_file_name)
    plt.show()


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
    intrusion_interest=INTRUSION_INTEREST,
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
        if intrusion_radius not in intrusion_interest:
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


def average_culling_cause_statistics(
    df,
    group_by="intrusion_radius",
    max_steps=10000,
    is_save=False,
    image_folder_path="",
    intrusion_interest=INTRUSION_INTEREST,
):
    # Merge the culling statistics with the configuration data to get the intrusion radius
    # Group the data by the specified column and calculate the mean of the second column

    grouped_data = df.groupby(group_by).agg(
        {
            "culling_statistics": ["mean"],
            "death_by_culling": ["mean"],
            "death_by_age": ["mean"],
            "death_by_predator": ["mean"],
            "death_by_starvation": ["mean"],
        }
    )

    average_culling_statistic = grouped_data["culling_statistics"]["mean"]
    average_death_by_culling = grouped_data["death_by_culling"]["mean"]
    average_death_by_predator = grouped_data["death_by_predator"]["mean"]
    average_death_by_starvation = grouped_data["death_by_starvation"]["mean"]
    average_death_by_age = grouped_data["death_by_age"]["mean"]

    label = convert_to_title_case(group_by)
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/average_culling_statistics_{group_by}.svg"
    )

    plt.figure(figsize=(12, 8))

    for intrusion_radius in average_culling_statistic.index:
        if intrusion_radius not in intrusion_interest:
            continue
        x_value = average_culling_statistic[intrusion_radius][:, 0]
        y_value = average_culling_statistic[intrusion_radius][:, 1]

        plt.plot(  # Plot the average culling statistics
            x_value,
            y_value,
            label=f"Culling Statistic - {label}: {intrusion_radius}",
        )

        x_culling = average_death_by_culling[intrusion_radius][:, 0]
        y_culling = average_death_by_culling[intrusion_radius][:, 1]
        plt.plot(
            x_culling,
            y_culling,
            label=f"Death by Culling - {label}: {intrusion_radius}",
        )

        x_predator = average_death_by_predator[intrusion_radius][:, 0]
        y_predator = average_death_by_predator[intrusion_radius][:, 1]
        plt.plot(
            x_predator,
            y_predator,
            label=f"Death by Predator - {label}: {intrusion_radius}",
        )

        x_starvation = average_death_by_starvation[intrusion_radius][:, 0]
        y_starvation = average_death_by_starvation[intrusion_radius][:, 1]
        plt.plot(
            x_starvation,
            y_starvation,
            label=f"Death by Starvation - {label}: {intrusion_radius}",
        )

        x_age = average_death_by_age[intrusion_radius][:, 0]
        y_age = average_death_by_age[intrusion_radius][:, 1]
        plt.plot(
            x_age,
            y_age,
            label=f"Death by Age - {label}: {intrusion_radius}",
        )

    plt.xlabel(label)
    plt.ylabel("Average Culling Statistic")
    plt.title(f"Average Culling Cause Statistic by {label}")
    plt.legend()
    plt.grid(True)

    if is_save:
        plt.savefig(save_file_name)
    plt.show()


def create_predator_death_by_age(
    df,
    group_by="intrusion_radius",
    max_steps=10000,
    is_save=False,
    image_folder_path="",
    intrusion_interest=INTRUSION_INTEREST,
):

    grouped_data = df.groupby(group_by).agg(
        {
            "predator_death_by_age": ["mean"],
            "predator_death_by_starvation": ["mean"],
        }
    )

    average_predator_death_by_age = grouped_data["predator_death_by_age"]["mean"]
    average_predator_death_by_starvation = grouped_data["predator_death_by_starvation"][
        "mean"
    ]

    label = convert_to_title_case(group_by)
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/average_predator_death_{group_by}.svg"
    )

    plt.figure(figsize=(12, 8))

    # for intrusion_radius in average_predator_death_by_age.index:
    #     if intrusion_radius not in intrusion_interest:
    #         continue
    #     x_age_value = average_predator_death_by_age[intrusion_radius][:, 0]
    #     y_age_value = average_predator_death_by_age[intrusion_radius][:, 1]

    #     plt.plot(  # Plot the average culling statistics
    #         x_age_value,
    #         y_age_value,
    #         label=f"Old age - {label}: {intrusion_radius}",
    #     )

    for intrusion_radius in average_predator_death_by_starvation.index:
        if intrusion_radius not in intrusion_interest:
            continue
        x_starvation_value = average_predator_death_by_starvation[intrusion_radius][
            :, 0
        ]
        y_starvation_value = average_predator_death_by_starvation[intrusion_radius][
            :, 1
        ]

        plt.plot(  # Plot the average culling statistics
            x_starvation_value,
            y_starvation_value,
            label=f"Starvation- {label}: {intrusion_radius}",
        )

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )
    # plt.yscale("log")
    plt.title("Predator cause of death")
    plt.xlabel("Time Step")
    plt.ylabel("Total amount")
    plt.legend()
    if is_save:
        plt.savefig(save_file_name)

    plt.show()


def create_culling_drop_scatter_plot(
    df,
    group_by="intrusion_radius",
    max_steps=10000,
    is_save=False,
    image_folder_path="",
    intrusion_interest=INTRUSION_INTEREST,
):

    grouped_data = df.groupby(group_by).agg(
        {
            "culling_statistics": ["mean"],
        }
    )
    average_culling_cause_statistics = grouped_data["culling_statistics"]["mean"]

    culling_drop_percentages = []
    intrusion_radii = []

    for intrusion_radius in average_culling_cause_statistics.index:
        if intrusion_radius not in intrusion_interest:
            continue
        culling_statistics = average_culling_cause_statistics[intrusion_radius]
        half_steps = int(len(culling_statistics) / 2)
        second_half_culling = culling_statistics[half_steps:]
        mean_second_half_culling = np.mean(second_half_culling, axis=0)

        culling_drop_percentages.append(mean_second_half_culling[1])
        intrusion_radii.append(intrusion_radius)

    # Create the scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(
        intrusion_radii, culling_drop_percentages, color="blue", label="Culling Drop"
    )
    plt.xlabel("Intrusion Radius")
    plt.ylabel("Mean Culling in Second Half")
    plt.title("Mean Culling in Second Half vs Intrusion Radius")
    plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if is_save:
        plt.savefig(image_folder_path + "culling_drop_vs_intrusion_radius.png")

    plt.show()

    # for k, folder_names_current in enumerate(FOLDER_NAMES_LIST):
    #     # Calculate average culling for the last 5000 steps
    #     total_culling_last_5000 = []
    #     intrusion_radius = None
    #     for folder_name in folder_names_current:
    #         culling_statistics = genfromtxt(
    #             f"{ROOT_PATH}{folder_name}/culling_statistics.csv", delimiter=","
    #         )
    #         half_steps = int(len(culling_statistics) / 2)
    #         avg_culling = np.sum(culling_statistics[-half_steps:, 1])
    #         total_culling_last_5000.append(avg_culling)

    #         # Get the intrusion radius
    #         config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
    #         intrusion_radius = config["intrusion"]["radius"]

    #     # Average across seeds
    #     avg_culling_with_intrusion = np.sum(total_culling_last_5000)

    #     if k == 0:
    #         baseline_culling = avg_culling_with_intrusion  # Intrusion radius = 0

    #     culling_drop_percent = (
    #         (baseline_culling - avg_culling_with_intrusion) / baseline_culling
    #     ) * 100
    #     culling_drop_percentages.append(culling_drop_percent)
    #     intrusion_radii.append(intrusion_radius)


if __name__ == "__main__":
    df = read_results(FOLDER_NAMES, ROOT_PATH, group_by="intrusion.radius")
    intrusion_interest = [0, 60]

    average_population_dynamics_2axis(
        df,
        group_by="intrusion_radius",
        is_save=True,
        only_reindeer=False,
        intrusion_interest=intrusion_interest,
        image_folder_path=IMAGE_FOLDER_NAME,
    )
    # average_culling_statistics(
    #     df,
    #     group_by="intrusion_radius",
    #     is_save=True,
    #     intrusion_interest=intrusion_interest,
    # )  # Haven't Average Culling Statistics ye

    # average_population_dynamics(
    #     df,
    #     group_by="food_regeneration_rate",
    #     is_save=True,
    # )
    # error_bar_population_dynamics(df, group_by="intrusion_radius", is_save=True)

    # create_predator_death_by_age(df, group_by="intrusion_radius", is_save=True)
    # intrusion_interest = [
    #     70,
    # ]
    average_culling_cause_statistics(
        df,
        group_by="intrusion_radius",
        is_save=True,
        intrusion_interest=intrusion_interest,
    )
