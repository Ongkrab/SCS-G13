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
IMAGE_FOLDER_NAME = "./merges/images"
FIGSIZE = (6, 3)
IS_EXPORT_WEB = True
image_folder_path = IMAGE_FOLDER_NAME


def read_population(result_list):
    result = pd.DataFrame()
    for result_file in result_list:
        file_path = ROOT_PATH + FOOD_REGENERATION_PATH + "/" + result_file
        df = pd.read_csv(file_path)

        column_name = result_file.split("RADIUS_")[1].split("_")[0]
        result[column_name] = df["average"]
    return result


def read_single_population(result_list, seed):
    result = pd.DataFrame()
    for result_file in result_list:
        file_path = ROOT_PATH + FOOD_REGENERATION_PATH + "/" + result_file
        df = pd.read_csv(file_path)

        column_name = result_file.split("RADIUS_")[1].split("_")[0]
        print(df.columns)
        if seed in df.columns:
            result[column_name] = df[seed]
        # result[column_name] = df["average"]
    print(result)
    return result


def find_difference_based_on_radius(df_culling):
    df_based = df_culling.iloc[0]
    result_temp = pd.DataFrame(columns=["radius", "average", "std"])
    based_average = 0
    temp = []
    for column in df_based.index:

        if column != "radius":
            temp.append(df_based[column])

    based_average = np.average(temp)
    based_std = np.std(temp)

    for index, row in df_culling.iterrows():
        new_row = row.astype(float) - df_based.astype(float)
        new_row["radius"] = row["radius"]
        temp_difference = []
        for col in df_culling.columns:
            if col != "radius":
                new_row[col] = row[col] - df_based[col]
                temp_difference.append(row[col] - df_based[col])
        print(len(temp_difference))
        average = np.average(temp_difference)
        std = np.std(temp_difference)
        min = np.min(temp_difference)
        max = np.max(temp_difference)
        average_percentage = (average / based_average) * 100
        std_percentage = (std / based_average) * 100
        new_row_df = {
            "radius": row["radius"],
            "average": average,
            "average_percentage": average_percentage,
            "std": std,
            "std_percentage": std_percentage,
            "min": min,
            "max": max,
        }
        result_temp = result_temp._append(new_row_df, ignore_index=True)

    print(result_temp)

    return result_temp


def read_culling(result_list):

    result = pd.DataFrame(columns=["radius", "average", "std"])
    df_sum_5000 = pd.DataFrame()
    for result_file in result_list:
        file_path = ROOT_PATH + FOOD_REGENERATION_PATH + "/" + result_file
        df = pd.read_csv(file_path)

        radius = result_file.split("RADIUS_")[1].split("_")[0]
        total_row = df.shape[0]
        start_row = round(total_row / 2)
        total_culling_5000 = []
        df_sum_5000_temp = {}
        for seed_column in df.columns:
            if seed_column.startswith("SEED"):
                culling_5000 = np.sum(df[seed_column][start_row:])
                total_culling_5000.append(culling_5000)
                df_sum_5000_temp[seed_column] = culling_5000

        df_sum_5000_temp["radius"] = radius
        if df_sum_5000.shape[0] == 0:
            df_sum_5000 = pd.DataFrame(df_sum_5000_temp, index=[0])
            df_sum_5000.set_index("radius")
        else:
            df_sum_5000 = df_sum_5000._append(
                pd.DataFrame(df_sum_5000_temp, index=[0]), ignore_index=False
            )

        average = np.average(total_culling_5000)
        std = np.std(total_culling_5000)
        new_row = {"radius": radius, "average": average, "std": std}
        result = result._append(new_row, ignore_index=True)
    based_average = result["average"][0]
    based_std = result["std"][0]
    result["diff_based"] = result["average"] - based_average

    result["diff_std_percentage"] = (
        (result["std"] - based_std) / based_average * 100
    ).abs()
    result["diff_percentage"] = (result["diff_based"] / based_average * 100).abs()
    print(result)

    find_difference_based_on_radius(df_sum_5000)
    return result


def single_population_dynamics_2axis(
    df_reindeer,
    df_predator,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/fig2_100_single_pop_comparison.svg"
    )
    # Plot the population dynamics
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

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

    ax1.set_ylim(0, 400)
    ax2.set_ylim(0, 80)
    ax1.set_xlim(0, 10000)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Reindeer Population", color="tab:blue")
    ax2.set_ylabel("Predator Population", color="tab:red")
    ax2.set_ylabel("Predator Population")

    box_to_anchor = (1.1, 1.05)
    # ax1.legend(loc="lower left", bbox_to_anchor=box_to_anchor)
    # ax2.legend(loc="lower right", bbox_to_anchor=box_to_anchor)
    ax1.legend(loc="upper left", fontsize="medium")
    ax2.legend(loc="upper right", fontsize="medium")
    # plt.legend(loc="upper right", fontsize="medium")

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    # plt.xlabel("Time Step", fontsize="large")
    plt.title("Population Dynamics for Intrusion Radius: 60")
    plt.tight_layout()
    plt.savefig(
        save_file_name,
        dpi=600,
    )

    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()


def single_population_dynamics_2axis_006(
    df_reindeer,
    df_predator,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/fig7_100_006_single_pop_comparison.svg"
    )
    # Plot the population dynamics
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

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

    ax1.set_ylim(0, 1000)
    ax2.set_ylim(0, 100)
    ax1.set_xlim(0, 10000)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Reindeer Population", color="tab:blue")
    ax2.set_ylabel("Predator Population", color="tab:red")
    ax2.set_ylabel("Predator Population")

    box_to_anchor = (1.1, 1.05)
    # ax1.legend(loc="lower left", bbox_to_anchor=box_to_anchor)
    # ax2.legend(loc="lower right", bbox_to_anchor=box_to_anchor)
    ax1.legend(loc="upper left", fontsize="medium")
    ax2.legend(loc="upper right", fontsize="medium")
    # plt.legend(loc="upper right", fontsize="medium")

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    # plt.xlabel("Time Step", fontsize="large")
    plt.title("Unconstraint Population Dynamics")
    plt.tight_layout()
    plt.savefig(
        save_file_name,
        dpi=600,
    )
    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()


def average_population_dynamics(
    df,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/fig4_100_average_pop_comparison.svg"
    )

    # Plot the population dynamics
    plt.figure(figsize=FIGSIZE)

    for intrusion_radius in intrusion_interest:
        plt.plot(
            df[f"{intrusion_radius}"],
            label=f"Prey - Radius: {intrusion_radius}",
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
    plt.title("Average Population Dynamics")
    plt.xlim(0, 10000)

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_file_name,
        dpi=600,
    )
    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()


def average_population_dynamics_2axis(
    df_reindeer,
    df_predator,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
    export_web=False,
):
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/fig3_100_average_population_60.svg"
    )

    # Plot the population dynamics
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

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

    ax1.set_ylim(0, 400)
    ax2.set_ylim(0, 80)
    ax1.set_xlim(0, 10000)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Reindeer Population", color="tab:blue")
    ax2.set_ylabel("Predator Population", color="tab:red")
    ax2.set_ylabel("Predator Population")

    box_to_anchor = (1.1, 1.05)
    # ax1.legend(loc="lower left", bbox_to_anchor=box_to_anchor)
    # ax2.legend(loc="lower right", bbox_to_anchor=box_to_anchor)
    ax1.legend(loc="upper left", fontsize="medium")
    ax2.legend(loc="upper right", fontsize="medium")

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

    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()


def create_culling_drop_scatter_plot(
    df_culling1,
    df_culling2,
    image_folder_path=IMAGE_FOLDER_NAME,
    intrusion_interest=INTRUSION_INTEREST,
):
    save_file_name = f"{ROOT_PATH}{image_folder_path}/fig5_100_culling_drop.svg"
    """
    Creates a scatter plot showing percentage drop in culling vs. intrusion radius.
    """
    culling_drop_percentages = []
    culling_drop_errors = []
    decreased_area_percentages = []

    intrusion_radii = []

    total_area = 100 * 300  # Total area of the simulation
    print(df_culling1)
    print("-----Food Regeneration 0.0025-----")
    print(df_culling2)
    print("-----Food Regeneration 0.0035-----")

    decreased_area_percentages = (
        0.5 * np.pi * df_culling1["radius"].astype(int) ** 2 / total_area
    ) * 100
    decreased_area_percentages = decreased_area_percentages.tolist()[1:]
    # print(decreased_area_percentages)

    df_culling1["std_percetage"] = df_culling1["std"] / df_culling1["average"] * 100
    # culling_drop_percentages = df_culling1["diff_percentage"].abs().tolist()
    # decreased_area_percentages = df_culling1["decresed_area_percent"].tolist()
    intrusion_radii = df_culling1["radius"].astype(str).tolist()
    intrusion_radii = intrusion_radii[1:]
    plt.figure(figsize=FIGSIZE)
    plt.bar(
        df_culling2["radius"].tolist()[1:],
        df_culling2["diff_percentage"].abs().tolist()[1:],
        color="tomato",
        width=0.4,
        label="Culling Drop (%)",
    )
    plt.errorbar(
        df_culling2["radius"].tolist()[1:],
        df_culling2["diff_percentage"].abs().tolist()[1:],
        df_culling2["diff_std_percentage"].tolist()[1:],
        capsize=4,
        color="orange",
        label="Culling Drop STD (%)",
        fmt=".",
    )
    plt.scatter(
        intrusion_radii,
        decreased_area_percentages,
        color="darkblue",
        label="Decreased Area (%)",
    )

    plt.xlabel("Intrusion Radius")
    plt.ylabel("Decreased Percentage (%)")
    plt.ylim(0, 60)
    plt.title("Culling Drop and Decreased Area vs Intrusion Radius")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(intrusion_radii)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_file_name,
        dpi=600,
    )

    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()

    # fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # ax2 = ax1.twinx()

    # # ax1.bar(
    # #     intrusion_radii,
    # #     culling_drop_percentages,
    # #     color="blue",
    # #     width=0.3,
    # #     label="Culling Drop (%)",
    # # )
    # ax1.errorbar(
    #     df_culling1["radius"].tolist()[1:],
    #     df_culling1["diff_percentage"].abs().tolist()[1:],
    #     df_culling1["diff_std_percentage"].tolist()[1:],
    #     capsize=4,
    #     color="grey",
    #     label="Culling Drop (%) - Food Regeneration 0.0025",
    #     # fmt="o",
    # )
    # ax1.errorbar(
    #     df_culling2["radius"].tolist()[1:],
    #     df_culling2["diff_percentage"].tolist()[1:],
    #     df_culling2["diff_std_percentage"].tolist()[1:],
    #     capsize=4,
    #     color="blue",
    #     label="Culling Drop (%) - Food Regeneration 0.0035",
    #     # fmt="o",
    # )
    # ax2.scatter(
    #     intrusion_radii,
    #     decreased_area_percentages,
    #     color="orange",
    #     label="Decreased Area (%)",
    # )

    # ax1.legend(loc="upper left", fontsize="large")
    # ax2.legend(loc="upper right", fontsize="large")
    # ax1.set_xticks(intrusion_radii)
    # # ax1.set_yscale("log")
    # # ax2.set_yscale("log")
    # ax1.set_xlabel("Intrusion Radius")
    # ax1.set_ylabel("Average Culling Drop")
    # ax1.set_ylim(0, 80)
    # ax2.set_ylabel("Decreased Percentage (%)")
    # ax2.set_ylim(0, 100)
    save_file_name = (
        f"{ROOT_PATH}{image_folder_path}/fig6_100_culling_drop_0025_0035.svg"
    )
    plt.figure(figsize=FIGSIZE)

    plt.errorbar(
        df_culling1["radius"].tolist()[1:],
        df_culling1["diff_percentage"].abs().tolist()[1:],
        df_culling1["diff_std_percentage"].tolist()[1:],
        capsize=4,
        color="red",
        label="Culling Drop STD (%) - Food Regeneration 0.0025",
        # fmt="o",
    )
    plt.errorbar(
        df_culling2["radius"].tolist()[1:],
        df_culling2["diff_percentage"].tolist()[1:],
        df_culling2["diff_std_percentage"].tolist()[1:],
        capsize=4,
        color="orange",
        label="Culling Drop STD (%) - Food Regeneration 0.0035",
        # fmt="o",
    )
    plt.scatter(
        intrusion_radii,
        decreased_area_percentages,
        color="darkblue",
        label="Decreased Area (%)",
    )

    # ax1.legend(loc="upper left", fontsize="large")
    # ax2.legend(loc="upper right", fontsize="large")
    plt.xticks(intrusion_radii)
    # ax1.set_yscale("log")
    # ax2.set_yscale("log")
    plt.xlabel("Intrusion Radius")
    plt.ylabel("Decreased Percentage (%)")
    # plt.ylim(0, 80)

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    # plt.xticks(intrusion_radii)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    plt.title("Culling Drop and Decreased Area vs Intrusion Radius")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        save_file_name,
        dpi=600,
    )

    if IS_EXPORT_WEB:
        save_file_html = f"{save_file_name}.html"

        fig = mpl_to_plotly(plt.gcf())
        pio.write_html(fig, save_file_html)

    plt.show()


if __name__ == "__main__":
    # Read the data from the specified folder
    # result_list = REINDEER_POPULATION_0_0025
    # INTRUSION_INTEREST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    df_reindeer = read_population(REINDEER_POPULATION_0_0025)
    df_predator = read_population(PREDATOR_POPULATION_0_0025)
    df_single_reindeer = read_single_population(REINDEER_POPULATION_0_0025, "SEED_113")
    df_single_predator = read_single_population(PREDATOR_POPULATION_0_0025, "SEED_113")

    INTRUSION_INTEREST = [0, 60]
    single_population_dynamics_2axis(
        df_single_reindeer,
        df_single_predator,
        intrusion_interest=INTRUSION_INTEREST,
        image_folder_path=IMAGE_FOLDER_NAME,
    )

    INTRUSION_INTEREST = [0, 30, 60, 90]
    average_population_dynamics(
        df_reindeer,
        intrusion_interest=INTRUSION_INTEREST,
        image_folder_path=IMAGE_FOLDER_NAME,
    )

    INTRUSION_INTEREST = [0, 60]
    average_population_dynamics_2axis(
        df_reindeer,
        df_predator,
        intrusion_interest=INTRUSION_INTEREST,
        image_folder_path=IMAGE_FOLDER_NAME,
    )

    df_culling_0025 = read_culling(CULLING_RATE_0_0025)
    FOOD_REGENERATION_PATH = "FOOD_REGENERATION_RATE_0_35"
    df_culling_0035 = read_culling(CULLING_RATE_0_0035)
    create_culling_drop_scatter_plot(
        df_culling_0025, df_culling_0035, image_folder_path=IMAGE_FOLDER_NAME
    )

    FOOD_REGENERATION_PATH = "CULLING_006"
    df_single_reindeer = read_single_population(REINDEER_POPULATION_0_0025, "SEED_005")
    df_single_predator = read_single_population(PREDATOR_POPULATION_0_0025, "SEED_005")

    INTRUSION_INTEREST = [0, 60]
    single_population_dynamics_2axis_006(
        df_single_reindeer,
        df_single_predator,
        intrusion_interest=INTRUSION_INTEREST,
        image_folder_path=IMAGE_FOLDER_NAME,
    )
