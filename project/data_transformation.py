import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os
from data_results import *

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
INTRUSION_INTEREST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def folder_names(root_path):
    return os.listdir(root_path)


def read_and_average_data(food_radius_path, average_file_name):
    """Read files from the specified folder and average the data in each file."""
    files = []
    seed_folders = []
    folder_path = os.path.join(ROOT_PATH, food_radius_path)
    for f in os.listdir(folder_path):
        if f.startswith("."):
            continue
        for file_name in os.listdir(os.path.join(folder_path, f)):
            if file_name.endswith(average_file_name):
                data_path = os.path.join(folder_path, f, file_name)
                files.append(data_path)

    # Initialize an empty list to store DataFrames
    data_frames = []

    # Read each file and append the DataFrame to the list
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        upper_folder_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file, header=None)
        df.columns = [upper_folder_name]
        data_frames.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(data_frames, axis=1)

    summary_ave_data = merged_df.copy()
    summary_ave_data["average"] = summary_ave_data.mean(numeric_only=True, axis=1)
    summary_ave_data["std"] = summary_ave_data.std(numeric_only=True, axis=1)

    return summary_ave_data


def read_and_average_first_column_data(food_radius_path, average_file_name):
    """Read files from the specified folder and average the data in each file."""
    files = []
    seed_folders = []
    folder_path = os.path.join(ROOT_PATH, food_radius_path)
    for f in os.listdir(folder_path):
        if f.startswith("."):
            continue
        for file_name in os.listdir(os.path.join(folder_path, f)):
            if file_name.endswith(average_file_name):
                data_path = os.path.join(folder_path, f, file_name)
                files.append(data_path)

    # Initialize an empty list to store DataFrames
    data_frames = []

    # Read each file and append the DataFrame to the list
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        upper_folder_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file, header=None)
        df.columns = ["reference", upper_folder_name]
        data_frames.append(df)

    # Concatenate all DataFrames, merging on the first column
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on="reference", how="outer")

    summary_ave_data = merged_df.copy()
    summary_ave_data["average"] = summary_ave_data.mean(numeric_only=True, axis=1)
    summary_ave_data["std"] = summary_ave_data.std(numeric_only=True, axis=1)

    return summary_ave_data


def save_averaged_data(datafram, file_name, root_path):
    print(f"Saving data to {root_path}{file_name}")
    os.makedirs(root_path, exist_ok=True)
    datafram.to_csv(f"{root_path}{file_name}")


if __name__ == "__main__":
    food_regeneration_rate = "0_35"
    folder_name = FOOD_REGENERATION_RATE_0_0035_PATH
    results_average_path = (
        f"{ROOT_PATH}/FOOD_REGENERATION_RATE_{food_regeneration_rate}/"
    )
    os.makedirs(results_average_path, exist_ok=True)
    average_data = "predator_population.csv"
    # average_data = "reindeer_population.csv"

    for food_radius_path in folder_name:
        df_average = read_and_average_data(food_radius_path, average_data)
        average_path = food_radius_path + "_average_" + average_data
        print(average_path)
        save_averaged_data(df_average, average_path, results_average_path)

    # average_data = "culling_statistics.csv"

    # for food_radius_path in folder_name:
    #     df_average = read_and_average_first_column_data(food_radius_path, average_data)
    #     average_path = food_radius_path + "_average_" + average_data
    #     print(average_path)
    #     save_averaged_data(df_average, average_path, results_average_path)

    # df = read_results(FOLDER_NAMES, ROOT_PATH, group_by="intrusion.radius")
