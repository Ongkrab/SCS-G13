import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import helper
import os

ROOT_PATH = "./results/"
CONFIG_PATH = "./config.json"
ROOT_PATH3 = "./results/"

FOLDER_NAMES_LIST1 = [
    [
        "seed1_intrusion0",
        "seed2_intrusion0",
        "seed3_intrusion0",
        "seed4_intrusion0",
        "seed5_intrusion0",
    ],
    [
        "seed1_intrusion40",
        "seed2_intrusion40",
        "seed3_intrusion40",
        "seed4_intrusion40",
        "seed5_intrusion40",
    ],
    [
        "seed1_intrusion50",
        "seed2_intrusion50",
        "seed3_intrusion50",
        "seed4_intrusion50",
        "seed5_intrusion50",
    ],
    [
        "seed1_intrusion60",
        "seed2_intrusion60",
        "seed3_intrusion60",
        "seed4_intrusion60",
        "seed5_intrusion60",
    ],
    [
        "seed1_intrusion70",
        "seed2_intrusion70",
        "seed3_intrusion70",
        "seed4_intrusion70",
        "seed5_intrusion70",
    ],
    [
        "seed1_intrusion80",
        "seed2_intrusion80",
        "seed3_intrusion80",
        "seed4_intrusion80",
        "seed5_intrusion80",
    ],
]

FOLDER_NAMES_LIST2 = [
    [
        "seed1_intrusion0",
        "seed2_intrusion0",
        "seed3_intrusion0",
        "seed4_intrusion0",
        "seed5_intrusion0",
    ],
    [
        "seed1_intrusion20",
        "seed2_intrusion20",
        "seed3_intrusion20",
        "seed4_intrusion20",
        "seed5_intrusion20",
    ],
    [
        "seed1_intrusion40",
        "seed2_intrusion40",
        "seed3_intrusion40",
        "seed4_intrusion40",
        "seed5_intrusion40",
    ],
    [
        "seed1_intrusion50",
        "seed2_intrusion50",
        "seed3_intrusion50",
        "seed4_intrusion50",
        "seed5_intrusion50",
    ],
    [
        "seed1_intrusion60",
        "seed2_intrusion60",
        "seed3_intrusion60",
        "seed4_intrusion60",
        "seed5_intrusion60",
    ],
    [
        "seed1_intrusion70",
        "seed2_intrusion70",
        "seed3_intrusion70",
        "seed4_intrusion70",
        "seed5_intrusion70",
    ],
    [
        "seed1_intrusion80",
        "seed2_intrusion80",
        "seed3_intrusion80",
        "seed4_intrusion80",
        "seed5_intrusion80",
    ],
]

FOLDER_NAMES_LIST3 = (
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
)

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
        reindeer_population = None
        predator_population = None
        for i, folder_name in enumerate(folder_names_current):
            reindeer_population_temp = genfromtxt(
                f"{ROOT_PATH}{folder_name}/reindeer_population.csv", delimiter=","
            )
            if reindeer_population is None:
                reindeer_population = np.zeros_like(reindeer_population_temp)
            reindeer_population += reindeer_population_temp
            predator_population_temp = genfromtxt(
                f"{ROOT_PATH}{folder_name}/predator_population.csv", delimiter=","
            )
            if predator_population is None:
                predator_population = np.zeros_like(predator_population_temp)
            predator_population += predator_population_temp

            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
            intrusion_radius = config["intrusion"]["radius"]
        reindeer_population /= len(folder_names_current)
        predator_population /= len(folder_names_current)
        plt.plot(
            reindeer_population,
            label=f"Reindeer Population - Intrusion Radius: {intrusion_radius}",
        )
        plt.plot(
            predator_population,
            label=f"Predator Population - Intrusion Radius: {intrusion_radius}",
        )

    plt.axvline(
        x=max_steps / 2,
        color="grey",
        linestyle="--",
        linewidth=2,
        label="Intrusion added",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title(
        f"Average Population Dynamics over {len(FOLDER_NAMES_LIST[0])} simulation pairs"
    )
    plt.legend()
    plt.tight_layout()  # rect=[1, 1, 0, 0])
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
            culling_statistics_temp = genfromtxt(
                f"{ROOT_PATH}{folder_name}/culling_statistics.csv", delimiter=","
            )
            if culling_statistics is None:
                culling_statistics = np.zeros_like(culling_statistics_temp)
            culling_statistics += culling_statistics_temp
            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
            intrusion_radius = config["intrusion"]["radius"]
        culling_statistics /= len(folder_names_current)
        plt.plot(
            culling_statistics[:, 0],
            culling_statistics[:, 1],
            label=f"Culling Statistics - Intrusion Radius: {intrusion_radius}",
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
    plt.title(
        f"Average Culling Statistics over {len(FOLDER_NAMES_LIST[0])} simulation pairs"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    death_by_culling_list = []
    death_by_starvation_list = []
    death_by_predator_list = []
    death_by_age_list = []
    intrusion_radii = []
    for folder_names_current in FOLDER_NAMES_LIST:
        sum_death_by_culling=0
        sum_death_by_starvation=0
        sum_death_by_predator=0
        sum_death_by_age=0
        for i, folder_name in enumerate(folder_names_current):
            death_by_culling = genfromtxt(f"{ROOT_PATH}{folder_name}/death_by_culling.csv", delimiter=',')
            sum_death_by_culling+=death_by_culling[250][1]-death_by_culling[126][1] 
            death_by_starvation = genfromtxt(f"{ROOT_PATH}{folder_name}/death_by_starvation.csv", delimiter=',')
            sum_death_by_starvation+=death_by_starvation[10000][1]-death_by_starvation[5001][1]
            death_by_predator = genfromtxt(f"{ROOT_PATH}{folder_name}/death_by_predator.csv", delimiter=',')
            sum_death_by_predator+=death_by_predator[10000][1]-death_by_predator[5001][1] 
            death_by_age = genfromtxt(f"{ROOT_PATH}{folder_name}/death_by_age.csv", delimiter=',')
            sum_death_by_age+=death_by_age[250][1]-death_by_age[126][1] 
            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
        intrusion_radii.append(config["intrusion"]["radius"])
        sum_death_by_culling/=len(folder_names_current)
        death_by_culling_list.append(sum_death_by_culling)
        sum_death_by_starvation/=len(folder_names_current)
        death_by_starvation_list.append(sum_death_by_starvation)
        sum_death_by_predator/=len(folder_names_current)
        death_by_predator_list.append(sum_death_by_predator)
        sum_death_by_age/=len(folder_names_current)
        death_by_age_list.append(sum_death_by_age)
    death_by_culling_list = np.array(death_by_culling_list)
    death_by_starvation_list = np.array(death_by_starvation_list)
    death_by_predator_list = np.array(death_by_predator_list)
    death_by_age_list = np.array(death_by_age_list)
    death_total_list = death_by_culling_list + death_by_starvation_list + death_by_predator_list + death_by_age_list
    plt.figure(figsize=(10, 5))
    plt.scatter(intrusion_radii, death_by_culling_list/death_total_list*100)
    plt.scatter(intrusion_radii, death_by_starvation_list/death_total_list*100)
    plt.scatter(intrusion_radii, death_by_predator_list/death_total_list*100)
    plt.scatter(intrusion_radii, death_by_age_list/death_total_list*100)
    plt.xlabel("Intrusion Radius")
    plt.ylabel("Cause of Death percentage of Total Deaths")
    plt.title("Cause of Death Statistics for Different Radii")
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_culling_drop_scatter_plot(
    FOLDER_NAMES_LIST,
    ROOT_PATH,
    max_steps=10000,
    is_save=False,
    image_folder_path="",
):
    """
    Creates a scatter plot showing percentage drop in culling vs. intrusion radius.
    """
    culling_drop_percentages = []
    decreased_area_percentages = []
    intrusion_radii = []

    total_area = 100 * 300  # Total area of the simulation

    for k, folder_names_current in enumerate(FOLDER_NAMES_LIST):
        # Calculate average culling for the last 5000 steps
        total_culling_last_5000 = []
        intrusion_radius = None
        for folder_name in folder_names_current:
            culling_statistics = genfromtxt(
                f"{ROOT_PATH}{folder_name}/culling_statistics.csv", delimiter=","
            )
            half_steps = int(len(culling_statistics) / 2)
            avg_culling = np.sum(culling_statistics[-half_steps:, 1])
            total_culling_last_5000.append(avg_culling)

            # Get the intrusion radius
            config = helper.load_config(f"{ROOT_PATH}{folder_name}/config.json")
            intrusion_radius = config["intrusion"]["radius"]

        # Average across seeds
        avg_culling_with_intrusion = np.sum(total_culling_last_5000)
        std_culling_with_intrusion = np.std(total_culling_last_5000)
        std_culling_percentage = []
        if k == 0:
            baseline_culling = avg_culling_with_intrusion  # Intrusion radius = 0
        else:
            culling_drop_percent = (
                (baseline_culling - avg_culling_with_intrusion) / baseline_culling
            ) * 100
            culling_drop_percentages.append(culling_drop_percent)

            culling_drop_percent_std = (
                std_culling_with_intrusion / baseline_culling
            ) * 100
            std_culling_percentage.append(culling_drop_percent_std)

            intrusion_radii.append(intrusion_radius)

            # Calculate decreased area percentage
            decreased_area_percent = (
                0.5 * np.pi * intrusion_radius**2 / total_area
            ) * 100
            decreased_area_percentages.append(decreased_area_percent)
        print(intrusion_radii)
    # Create the scatter plot
    plt.figure(figsize=(10, 5))
    # Plot culling drop percentages
    # plt.errorbar(
    #     intrusion_radii,
    #     culling_drop_percentages,
    #     std_culling_percentage,
    #     color="blue",
    #     label="Culling Drop (%)",
    # )
    plt.bar(
        intrusion_radii,
        culling_drop_percentages,
        color="blue",
        width=2,
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

    if is_save:
        plt.savefig(image_folder_path + "culling_drop_vs_intrusion_radius.svg")

    plt.show()


if __name__ == "__main__":
    # create_population_dynamics_multi_run(FOLDER_NAMES_LIST1, ROOT_PATH)
    # create_culling_statistics_multi_run(FOLDER_NAMES_LIST1, ROOT_PATH)
    create_culling_drop_scatter_plot(
        FOLDER_NAMES_LIST3, ROOT_PATH3, image_folder_path="./images/", is_save=True
    )
