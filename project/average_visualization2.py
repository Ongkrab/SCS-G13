import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os
import plotly.tools as tools
from plotly.tools import mpl_to_plotly
import plotly.io as pio
from data_results import *

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
INTRUSION_INTEREST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

FOOD_REGENERATION_PATH = "FOOD_REGENERATION_RATE_0_25"
IMAGE_FOLDER_NAME = "merges/images"


def read_population(result_list):
    result = pd.DataFrame()
    for result_file in result_list:
        file_path = ROOT_PATH + FOOD_REGENERATION_PATH + "/" + result_file
        df = pd.read_csv(file_path)

        column_name = result_file.split("RADIUS_")[1].split("_")[0]
        result[column_name] = df["average"]
    return result


def read_culling(result_list):

    result = pd.DataFrame(columns=["radius", "average", "std"])
    for result_file in result_list:
        file_path = ROOT_PATH + FOOD_REGENERATION_PATH + "/" + result_file
        df = pd.read_csv(file_path)

        column_name = result_file.split("RADIUS_")[1].split("_")[0]
        total_row = df.shape[0]
        start_row = round(total_row / 2)
        total_culling_5000 = []
        for seed_column in df.columns:
            if seed_column.startswith("SEED"):
                culling_5000 = np.sum(df[seed_column][start_row:])
                total_culling_5000.append(culling_5000)
        average = np.average(total_culling_5000)
        std = np.std(total_culling_5000)
        new_row = {"radius": column_name, "average": average, "std": std}

        result = result._append(new_row, ignore_index=True)
    based_average = result["average"][0]
    result["diff_based"] = result["average"] - based_average
    result["diff_percentage"] = result["diff_based"] / based_average * 100
    return result


def average_population_dynamics(
    df,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    save_file_name = f"{ROOT_PATH}{image_folder_path}/average_population_dynamics.svg"
    # Plot the population dynamics
    plt.figure(figsize=(8, 4))

    for intrusion_radius in intrusion_interest:
        plt.plot(
            df[f"{intrusion_radius}"],
            label=f"Prey Population - intrusion radius: {intrusion_radius}",
        )

    max_steps = df.shape[0]
    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Average Population Dynamics over 100 simulation pairs")
    plt.xlim(0, 10000)

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_file_name,
        dpi=600,
    )
    # if export_web:
    #     save_file_html = f"{ROOT_PATH}{image_folder_path}/average_population_dynamics_{group_by}.html"
    #     fig = mpl_to_plotly(plt.gcf())

    #     # Save as an interactive HTML file
    #     pio.write_html(fig, save_file_html)
    plt.show()


def average_population_dynamics_2axis(
    df_reindeer,
    df_predator,
    group_by="intrusion_radius",
    only_reindeer=False,
    is_save=False,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
    export_web=False,
):
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/population_dynamics_2axis_{group_by}.svg"
    )

    # Plot the population dynamics
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax2 = ax1.twinx()

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
    ]

    i = 0

    for intrusion_radius in intrusion_interest:
        column_name = f"{intrusion_radius}"
        ax1.plot(
            df_reindeer[column_name],
            # label=f"Prey Population - Intrusion Radius: {intrusion_radius}",
            label=f"Prey - Radius: {intrusion_radius}",
            color=colors[i],
        )

        ax2.plot(
            df_predator[column_name],
            label=f"Predator - Radius: {intrusion_radius}",
            color=colors[i + 1],
        )
        i += 2

    max_steps = df_reindeer.shape[0]

    ax1.set_ylim(0, 350)
    ax2.set_ylim(0, 50)
    ax1.set_xlim(0, 10000)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Reindeer Population", color="tab:blue")
    ax2.set_ylabel("Predator Population", color="tab:red")
    ax2.set_ylabel("Predator Population")

    box_to_anchor = (1.1, 1.05)
    # ax1.legend(loc="lower left", bbox_to_anchor=box_to_anchor)
    # ax2.legend(loc="lower right", bbox_to_anchor=box_to_anchor)
    ax1.legend(loc="upper left", fontsize="large")
    ax2.legend(loc="upper right", fontsize="large")

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    # plt.xlabel("Time Step", fontsize="large")
    plt.title("Average Population Dynamics for Intrusion Radius: 60")
    plt.tight_layout()
    plt.savefig(
        save_file_name,
        dpi=600,
    )
    plt.show()


def create_culling_drop_scatter_plot(
    df_culling,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    """
    Creates a scatter plot showing percentage drop in culling vs. intrusion radius.
    """
    culling_drop_percentages = []
    culling_drop_errors = []
    decreased_area_percentages = []

    intrusion_radii = []

    total_area = 100 * 300  # Total area of the simulation
    print(df_culling)

    df_culling["decresed_area_percent"] = (
        0.5 * np.pi * df_culling["radius"].astype(int) ** 2 / total_area
    ) * 100
    culling_drop_percentages = df_culling["diff_percentage"].abs().tolist()
    decreased_area_percentages = df_culling["decresed_area_percent"].tolist()
    intrusion_radii = df_culling["radius"].astype(str).tolist()

    plt.figure(figsize=(8, 4))

    plt.bar(
        intrusion_radii,
        culling_drop_percentages,
        color="blue",
        width=0.5,
        label="Culling Drop (%)",
    )
    plt.scatter(
        intrusion_radii,
        decreased_area_percentages,
        color="orange",
        label="Decreased Area (%)",
    )

    plt.xlabel("Intrusion Radius")
    plt.ylabel("Decreased Percentage (%)")
    plt.title("Culling Drop and Decreased Area vs Intrusion Radius")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(intrusion_radii)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    plt.legend()
    plt.tight_layout()

    # plt.savefig(image_folder_path + "culling_drop_vs_intrusion_radius.svg")
    plt.show()

    # culling_drop_percentages = df_culling["average"].tolist()
    # culling_errors = df_culling["std"].tolist()
    # fig, ax1 = plt.subplots(figsize=(8, 4))

    # ax2 = ax1.twinx()

    # plt.bar(
    #     intrusion_radii,
    #     culling_drop_percentages,
    #     color="blue",
    #     width=0.5,
    #     label="Culling Drop (%)",
    # )
    # plt.errorbar(
    #     intrusion_radii,
    #     culling_drop_percentages,
    #     culling_errors,
    #     color="black",
    #     fmt="o",
    # )
    # plt.scatter(
    #     intrusion_radii,
    #     decreased_area_percentages,
    #     color="orange",
    #     label="Decreased Area (%)",
    # )

    # plt.xlabel("Intrusion Radius")
    # plt.ylabel("Decreased Percentage (%)")
    # plt.title("Culling Drop and Decreased Area vs Intrusion Radius")
    # plt.grid(axis="y", linestyle="--", alpha=0.6)
    # plt.xticks(intrusion_radii)
    # plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    # plt.legend()
    # plt.tight_layout()

    # # plt.savefig(image_folder_path + "culling_drop_vs_intrusion_radius.svg")
    # plt.show()
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
    #     std_culling_with_intrusion = np.std(total_culling_last_5000)
    #     std_culling_percentage = []
    #     if k == 0:
    #         baseline_culling = avg_culling_with_intrusion  # Intrusion radius = 0
    #     else:
    #         culling_drop_percent = (
    #             (baseline_culling - avg_culling_with_intrusion) / baseline_culling
    #         ) * 100
    #         culling_drop_percentages.append(culling_drop_percent)

    #         culling_drop_percent_std = (
    #             std_culling_with_intrusion / baseline_culling
    #         ) * 100
    #         std_culling_percentage.append(culling_drop_percent_std)

    #         intrusion_radii.append(intrusion_radius)

    #         # Calculate decreased area percentage
    #         decreased_area_percent = (
    #             0.5 * np.pi * intrusion_radius**2 / total_area
    #         ) * 100
    #         decreased_area_percentages.append(decreased_area_percent)
    #     print(intrusion_radii)
    # Create the scatter plot
    # plt.figure(figsize=(8, 4))
    # Plot culling drop percentages
    # plt.errorbar(
    #     intrusion_radii,
    #     culling_drop_percentages,
    #     std_culling_percentage,
    #     color="blue",
    #     label="Culling Drop (%)",
    # )
    # plt.bar(
    #     intrusion_radii,
    #     culling_drop_percentages,
    #     color="blue",
    #     width=2,
    #     label="Culling Drop (%)",
    # )
    # plt.scatter(
    #     intrusion_radii,
    #     decreased_area_percentages,
    #     color="orange",
    #     label="Decreased Area (%)",
    # )

    # plt.xlabel("Intrusion Radius")
    # plt.ylabel("Decreased Percentage (%)")
    # plt.title("Culling Drop and Decreased Area vs Intrusion Radius")
    # plt.grid(axis="y", linestyle="--", alpha=0.6)
    # plt.xticks(intrusion_radii)
    # plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    # plt.legend()
    # plt.tight_layout()

    # if is_save:
    #     plt.savefig(image_folder_path + "culling_drop_vs_intrusion_radius.svg")

    # plt.show()


if __name__ == "__main__":
    # Read the data from the specified folder
    # result_list = REINDEER_POPULATION_0_0025
    # INTRUSION_INTEREST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    INTRUSION_INTEREST = [0, 30, 60, 90]

    df_reindeer = read_population(REINDEER_POPULATION_0_0025)
    df_predator = read_population(PREDATOR_POPULATION_0_0025)
    # average_population_dynamics(df_reindeer, intrusion_interest=INTRUSION_INTEREST)

    INTRUSION_INTEREST = [0, 60]
    # average_population_dynamics_2axis(
    #     df_reindeer, df_predator, intrusion_interest=INTRUSION_INTEREST
    # )

    df_culling = read_culling(CULLING_RATE_0_0025)
    create_culling_drop_scatter_plot(df_culling)
