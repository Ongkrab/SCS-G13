import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

folder_name = "20241206-133229"
result_folder_path = "./hello2/" + folder_name + "/"
image_folder_path = result_folder_path + "images/"

if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

reindeer_population = genfromtxt(
    result_folder_path + "reindeer_population.csv", delimiter=","
)
predator_population = genfromtxt(
    result_folder_path + "predator_population.csv", delimiter=","
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


# Plot population dynamics
plt.figure()
plt.plot(reindeer_population, label="Reindeer Population", color="blue")
plt.plot(predator_population, label="Predator Population", color="red")
plt.xlabel("Time Step")
plt.ylabel("Population")
plt.title("Population Dynamics")
plt.legend()
plt.savefig(image_folder_path + "population_dynamics.png")
plt.show()

death_by_age = np.array(death_by_age)
death_by_starvation = np.array(death_by_starvation)
death_by_predator = np.array(death_by_predator)
death_by_culling = np.array(death_by_culling)
plt.plot(death_by_age[:, 0], death_by_age[:, 1], color="blue", label="Old age")
plt.plot(
    death_by_starvation[:, 0], death_by_starvation[:, 1], color="green", label="Starved"
)
plt.plot(death_by_predator[:, 0], death_by_predator[:, 1], color="red", label="Eaten")
plt.plot(death_by_culling[:, 0], death_by_culling[:, 1], color="orange", label="Culled")
plt.title("Cause of death")
plt.xlabel("Time Step")
plt.ylabel("Total amount")
plt.legend()
plt.savefig(image_folder_path + "death_causes.png")
plt.show()

culling_statistics = np.array(culling_statistics)
plt.plot(culling_statistics[:, 0], culling_statistics[:, 1])
plt.title("Culling statistics")
plt.xlabel("Time Step")
plt.ylabel("Amount culled each season")
plt.savefig(image_folder_path + "culling_statistics.png")
plt.show()
