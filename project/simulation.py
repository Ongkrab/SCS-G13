import numpy as np
import time
from predator import Predator
from reindeer import Reindeer
import os
import visualization


# from matplotlib.colors import ListedColormap, BoundaryNorm
from helper import *
from animate import *


class Simulation:
    def __init__(
        self,
        config,
        seed_number,
        is_batch,
        result_path,
        isPlotResults=True,
        isAnimate=False,
        capture_interval=10000,
        max_steps=1000,
    ):
        self.config = config
        self.result_path = result_path
        self.is_batch = is_batch
        self.isPlotResults = isPlotResults
        self.seed_number = seed_number
        self.isAnimate = isAnimate
        self.capture_interval = capture_interval
        self.max_steps = max_steps
        self.initialize()

    def initialize(self):
        current_time = time.strftime("%Y%m%d-%H%M%S")

        print("Initializing simulation...")
        environment = self.config["environment"]
        intrusion = self.config["intrusion"]

        self.food_regeneration_rate = environment["food_regeneration_rate"]
        self.intrusion_radius = intrusion["radius"]

        image_folder = "images"
        main_path = self.result_path
        subfolder_path = (
            "FOOD_REGENERATION_RATE_"
            + str(self.food_regeneration_rate)
            + "INTRUSION_RADIUS_"
            + str(self.intrusion_radius)
            + "/"
        )
        subsubfolder_path = f"SEED_{self.seed_number:03}/"

        self.result_folder_path = main_path + subfolder_path + subsubfolder_path

        self.result_image_path = self.result_folder_path + image_folder + "/"

        # Create result folder if it doesn't exist

        if not os.path.exists(main_path):
            os.makedirs(main_path)

        if not os.path.exists(self.result_folder_path):
            os.makedirs(self.result_folder_path)

        if not os.path.exists(self.result_image_path):
            os.makedirs(self.result_image_path)

        # Data storage
        self.reindeer_population = []
        self.predator_population = []
        self.culling_statistics = []
        self.predator_reintroduction = []
        self.death_by_culling = [[0, 0]]
        self.death_by_age = [[0, 0]]
        self.death_by_predator = [[0, 0]]
        self.death_by_starvation = [[0, 0]]
        self.predator_death_by_age = [[0, 0]]
        self.predator_death_by_starvation = [[0, 0]]
        self.reindeer_clusterings_coefficient = []

    def simulate(self):
        simulation = self.config["simulation"]
        environment = self.config["environment"]
        reindeer = self.config["reindeer"]
        predator = self.config["predator"]
        intrusion = self.config["intrusion"]

        # Simulation parameters
        is_coeffient_calculate = simulation["is_coeffient_calculate"]

        # Environment parameters
        grid_size = tuple(environment["grid_size"])
        num_reindeer = environment["num_reindeer"]
        num_predators = environment["num_predators"]
        food_regeneration_rate = environment["food_regeneration_rate"]
        reproduction_interval = environment["reproduction_interval"]
        culling_rate = environment["culling_rate"]
        culling_threshold = environment["culling_threshold"]
        max_reindeer_population = environment["max_reindeer_population"]

        #############################
        ## Reindeer parameters
        #############################
        # Assign individual variables for reindeer parameters
        reindeer_initial_age = reindeer["initial"]["age"]
        reindeer_max_speed = reindeer["initial"]["max_speed"]
        reindeer_max_age = reindeer["initial"]["max_age"]
        reindeer_energy = reindeer["initial"]["energy"]
        reindeer_grazing_speed = reindeer["initial"]["grazing_speed"]
        reindeer_energy_decay = reindeer["initial"]["energy_decay"]

        ## Assign individual variables for reindeer reproduction parameters
        reindeer_reproductive_age = reindeer["reproduction"]["reproductive_age"]
        reindeer_reproduction_rate = reindeer["reproduction"]["reproduction_rate"]
        reindeer_reproduction_energy = reindeer["reproduction"]["reproduction_energy"]
        reindeer_reproduction_distance = reindeer["reproduction"][
            "reproduction_distance"
        ]
        reindeer_offspring_energy = reindeer["reproduction"]["offspring_energy"]

        # Assign individual variables for reindeer behavior parameters
        reindeer_protected_range = reindeer["behavior"]["protected_range"]
        reindeer_visual_range = reindeer["behavior"]["visual_range"]
        reindeer_alert_range = reindeer["behavior"]["alert_range"]
        reindeer_grazing_rate = reindeer["behavior"]["grazing_rate"]
        reindeer_energy_threshold = reindeer["behavior"]["energy_threshold"]

        #############################
        ## Predator parameters
        #############################

        # Assign individual variables for predator parameters
        predator_initial_age = predator["initial"]["age"]
        predator_max_age = predator["initial"]["max_age"]
        predator_energy = predator["initial"]["energy"]
        predator_energy_decay = predator["initial"]["energy_decay"]
        predator_cruise_speed = predator["initial"]["cruise_speed"]
        predator_hunt_speed = predator["initial"]["hunt_speed"]
        predator_energy_threshold = predator["initial"]["energy_threshold"]

        # Assign individual variables for predator reproduction parameters
        predator_reproductive_age = predator["reproduction"]["reproductive_age"]
        predator_reproduction_rate = predator["reproduction"]["reproduction_rate"]
        predator_reproduction_energy = predator["reproduction"]["reproduction_energy"]
        predator_offspring_energy = predator["reproduction"]["offspring_energy"]
        predator_reproduction_distance = predator["reproduction"][
            "reproduction_distance"
        ]

        # Assign individual variables for predator behavior parameters
        predator_visual_range = predator["behavior"]["visual_range"]
        predator_eating_range = predator["behavior"]["eating_range"]
        predator_energy_gain = predator["behavior"]["energy_gain"]

        print("Configuration loaded.")

        #############################
        ## Intrusion parameters
        #############################

        # Assign individual variables for intrusion parameters
        intrusion_center = None
        intrusion_radius = None

        #################################
        ## Create Environment
        #################################
        np.random.seed(self.seed_number)

        # Initialize food grid
        food_grid = np.random.uniform(0.2, 0.8, grid_size)
        set_grid_size(grid_size)

        # Initialize agents with random ages
        reindeers = [
            Reindeer(
                x=np.random.uniform(0, grid_size[0]),
                y=np.random.uniform(0, grid_size[1]),
                age=np.random.randint(0, 15),  # Random age between 0 and 15
                max_age=reindeer_max_age,
                energy=reindeer_energy,
                energy_decay=reindeer_energy_decay,
                reproductive_age=reindeer_reproductive_age,
                reproduction_rate=reindeer_reproduction_rate,
                reproduction_energy=reindeer_reproduction_energy,
                max_speed=reindeer_max_speed,
                grazing_speed=reindeer_grazing_speed,
                energy_threshold=reindeer_energy_threshold,
            )
            for _ in range(num_reindeer)
        ]

        predators = [
            Predator(
                x=np.random.uniform(0, grid_size[0]),
                y=np.random.uniform(0, grid_size[1]),
                age=np.random.randint(0, 8),  # Random age between 0 and 15
                max_age=predator_max_age,
                energy=predator_energy,
                energy_decay=predator_energy_decay,
                reproductive_age=predator_reproductive_age,
                reproduction_rate=predator_reproduction_rate,
                reproduction_energy=predator_reproduction_energy,
                cruise_speed=predator_cruise_speed,
                hunt_speed=predator_hunt_speed,
                energy_threshold=predator_energy_threshold,
            )
            for _ in range(num_predators)
        ]

        print("Create Environment: Complete")

        #################################
        ## Simulation Loop
        #################################
        startTime = time.time()
        fig = None
        if self.isAnimate:
            fig = plt.figure(figsize=(8, 4))

        # Simulation loop
        for step in range(self.max_steps):
            print(
                f"Food Regenerate = {self.food_regeneration_rate} and intrusion radius = {self.intrusion_radius}: Step {step} out of {self.max_steps}",
                end="\r",
            )
            if self.isAnimate:
                if stop_loop:  # Check the flag
                    print("Loop exited.")
                    break

            # Update food grid (simple logistic regeneration)
            food_grid += food_regeneration_rate * (1 - food_grid)
            food_grid = np.clip(
                food_grid, 0, 1
            )  # Ensure food values stay within [0, 1]

            # Move agents
            temp_death_by_starvation = self.death_by_starvation[-1][1]
            for reindeer in reindeers[:]:
                reindeer.move(
                    predators=predators,
                    food_grid=food_grid,
                    reindeers=reindeers,
                    protected_range=reindeer_protected_range,
                    visual_range=reindeer_visual_range,
                    alert_range=reindeer_alert_range,
                    exclusion_center=intrusion_center,
                    exclusion_radius=intrusion_radius,
                )  # Increased visual range
                reindeer.graze(
                    food_grid=food_grid, grazing_rate=reindeer_grazing_rate
                )  # Faster grazing
                if reindeer.energy <= 0:
                    reindeers.remove(reindeer)  # Remove dead reindeer
                    temp_death_by_starvation += 1

            self.death_by_starvation.append([step, temp_death_by_starvation])

            temp_death_by_predator = self.death_by_predator[-1][1]
            temp_number_of_reindeers = len(reindeers)
            temp_death_by_starvation = self.predator_death_by_starvation[-1][1]
            for predator in predators[:]:
                predator.move(
                    prey=reindeers,
                    visual_range=predator_visual_range,
                    exclusion_center=intrusion_center,
                    exclusion_radius=intrusion_radius,
                )  # Increased hunting range

                predator.eat(
                    prey=reindeers,
                    eating_range=predator_eating_range,
                    energy_gain=predator_energy_gain,
                )  # Higher energy gain from eating

                if predator.energy <= 0:
                    predators.remove(predator)  # Remove dead predators
                    temp_death_by_starvation += 1
            self.predator_death_by_starvation.append([step, temp_death_by_starvation])
            self.death_by_predator.append(
                [
                    step,
                    temp_death_by_predator + temp_number_of_reindeers - len(reindeers),
                ]
            )

            # Save population counts
            self.reindeer_population.append(len(reindeers))
            self.predator_population.append(len(predators))

            # Reproduction every reproduction_interval steps
            if step % reproduction_interval == 0:
                temp_death_by_age = self.death_by_age[-1][1]
                for reindeer in reindeers[:]:
                    reindeer.update_age()
                    if reindeer.age > reindeer.max_age:
                        reindeers.remove(reindeer)
                        temp_death_by_age += 1
                    if np.random.rand() < reindeer.reproduction_rate:
                        reindeer.reproduce(
                            reindeers=reindeers,
                            offspring_energy=reindeer_offspring_energy,
                            reproduction_distance=reindeer_reproduction_distance,
                            exclusion_center=intrusion_center,
                            exclusion_radius=intrusion_radius,
                        )
                self.death_by_age.append([step, temp_death_by_age])

                temp_death_by_age = self.predator_death_by_age[-1][1]
                for predator in predators[:]:
                    predator.update_age()
                    if predator.age > predator.max_age:
                        predators.remove(predator)
                        temp_death_by_age += 1
                    if np.random.rand() < predator.reproduction_rate:
                        predator.reproduce(
                            predators=predators,
                            offspring_energy=predator_offspring_energy,
                            reproduction_distance=predator_reproduction_distance,
                            exclusion_center=intrusion_center,
                            exclusion_radius=intrusion_radius,
                        )
                self.predator_death_by_age.append([step, temp_death_by_age])

                if len(predators) == 0:
                    self.predator_reintroduction.append([step, 0])
                    predators = [
                        Predator(
                            x=np.random.uniform(0, grid_size[0]),
                            y=np.random.uniform(0, grid_size[1]),
                            age=np.random.randint(0, 15),  # Random age between 0 and 15
                            max_age=predator_max_age,
                            energy=predator_energy,
                            energy_decay=predator_energy_decay,
                            reproductive_age=predator_reproductive_age,
                            reproduction_rate=predator_reproduction_rate,
                            reproduction_energy=predator_reproduction_energy,
                            cruise_speed=predator_cruise_speed,
                            hunt_speed=predator_hunt_speed,
                            energy_threshold=predator_energy_threshold,
                        )
                        for _ in range(num_predators)
                    ]

            if step % reproduction_interval == 0:
                # Culling of the herd
                if int(len(reindeers)) > culling_threshold:
                    num_to_remove = int(len(reindeers) * culling_rate)
                    if len(reindeers) - num_to_remove < culling_threshold:
                        num_to_remove = len(reindeers) - culling_threshold
                    if len(reindeers) > max_reindeer_population:
                        # num_to_remove = max(
                        #     int(len(reindeers) * culling_rate),
                        #     len(reindeers) - max_reindeer_population,
                        # )
                        num_to_remove = int(
                            ((len(reindeers) - max_reindeer_population) / 100 + 1)
                            * culling_rate
                            * len(reindeers)
                        )
                else:
                    num_to_remove = 0
                # Randomly select indices to remove using numpy
                indices_to_remove = np.random.choice(
                    len(reindeers), num_to_remove, replace=False
                )
                # Remove the selected objects from the list
                objects_to_remove = [reindeers[i] for i in indices_to_remove]
                reindeers = [
                    obj for i, obj in enumerate(reindeers) if i not in indices_to_remove
                ]
                self.death_by_culling.append(
                    [step, self.death_by_culling[-1][1] + num_to_remove]
                )
                self.culling_statistics.append([step, num_to_remove])

            # Intrusion logic
            if step == self.max_steps // 2:
                if not intrusion["center_relative_grid"] == None:
                    intrusion_center_relative = tuple(intrusion["center_relative_grid"])
                else:
                    intrusion_center_relative = None

                if intrusion_center_relative == None:
                    intrusion_center = None
                else:
                    intrusion_center = (
                        intrusion_center_relative[0] * grid_size[0],
                        intrusion_center_relative[1] * grid_size[1],
                    )

                intrusion_radius = intrusion["radius"]

            # Calcualte reindeer clustering
            reindeer_positions_x = np.array([r.get_position()[0] for r in reindeers])
            reindeer_positions_y = np.array([r.get_position()[1] for r in reindeers])

            if is_coeffient_calculate:
                clustering = global_clustering(
                    reindeer_positions_x,
                    reindeer_positions_y,
                    reindeer_protected_range,
                    grid_size[0],
                )
                self.reindeer_clusterings_coefficient.append(clustering)

            if not self.is_batch:

                # Plot the environment
                plot_simulation_step(
                    figures=fig,
                    grid_size=grid_size,
                    intrusion_center=intrusion_center,
                    intrusion_radius=intrusion_radius,
                    food_grid=food_grid,
                    result_image_path=self.result_image_path,
                    reindeers=reindeers,
                    predators=predators,
                    step=step,
                    isAnimate=self.isAnimate,
                    capture_interval=self.capture_interval,
                )

            # Stop if no reindeer are left
            if len(reindeers) == 0:
                print("All reindeer have been hunted.")
                break

        #############################
        ## End of simulation loop
        #############################
        endTime = time.time()
        difference = endTime - startTime
        minutes = difference // 60
        seconds = difference % 60
        print(
            f"Simulation finished. {endTime - startTime} seconds elapsed. ({minutes} minutes and {seconds} seconds) "
        )

    def save_result(self):
        print(
            f"Saving results for food regeneration rate = {self.food_regeneration_rate} and intrusion radius = {self.intrusion_radius}"
        )
        # Save the results
        reindeer_population = np.array(self.reindeer_population)
        reindeer_clusterings_coefficient = np.array(
            self.reindeer_clusterings_coefficient
        )
        death_by_age = np.array(self.death_by_age)
        death_by_starvation = np.array(self.death_by_starvation)
        death_by_predator = np.array(self.death_by_predator)
        death_by_culling = np.array(self.death_by_culling)
        culling_statistics = np.array(self.culling_statistics)
        predator_population = np.array(self.predator_population)
        predator_reintroduction = np.array(self.predator_reintroduction)
        predator_death_by_starvation = np.array(self.predator_death_by_starvation)
        predator_death_by_age = np.array(self.predator_death_by_age)

        # Save results to CSV files
        # config.save()
        with open(self.result_folder_path + "config.json", "w") as f:
            json.dump(self.config, f)

        np.savetxt(
            self.result_folder_path + "reindeer_population.csv",
            reindeer_population,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "predator_population.csv",
            predator_population,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "predator_reintroduction.csv",
            predator_reintroduction,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "death_by_age.csv", death_by_age, delimiter=","
        )
        np.savetxt(
            self.result_folder_path + "death_by_starvation.csv",
            death_by_starvation,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "death_by_predator.csv",
            death_by_predator,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "death_by_culling.csv",
            death_by_culling,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "culling_statistics.csv",
            culling_statistics,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "predator_death_by_age.csv",
            predator_death_by_age,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "predator_death_by_starvation.csv",
            predator_death_by_starvation,
            delimiter=",",
        )
        np.savetxt(
            self.result_folder_path + "reindeer_clustering_coefficient.csv",
            reindeer_clusterings_coefficient,
            delimiter=",",
        )

        if self.isPlotResults:
            visualization.visualize(
                root_path=self.result_path, folder_name=self.current_time
            )