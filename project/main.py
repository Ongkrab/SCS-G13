import numpy as np
import time
import matplotlib.pyplot as plt
from predator import Predator
from reindeer import Reindeer
from matplotlib.patches import Circle
#from matplotlib.colors import ListedColormap, BoundaryNorm
from helper import *
from pynput import keyboard

stop_loop = False

start_time = time.time()

# Function to handle key press events
def on_press(key):
    global stop_loop
    if key == keyboard.Key.esc:  # If Escape is pressed
        print("Escape key pressed! Stopping the loop...")
        stop_loop = True
        return False  # Stops the listener


listener = keyboard.Listener(on_press=on_press)
listener.start()


def main():
    # # Simulation parameters
    # grid_size = (200, 300)  # Height, Width
    # num_reindeer = 200  # Increased number of reindeer
    # num_predators = 5
    # max_steps = 10000
    # food_regeneration_rate = 0.003  # Slightly faster food regeneration
    # reproduction_interval = 35  # Reproduction occurs less frequently

    # #############################
    # ## Reindeer parameters
    # #############################
    # # Reindeer initial parameters
    # reindeer_initial_age = 0
    # reindeer_max_speed = 1.0
    # reindeer_max_age = 15
    # reindeer_energy = 1.0
    # reindeer_grazing_speed = 0.2
    # reindeer_energy_decay = 0.02

    # # Reindeer reproduction parameters
    # reindeer_reproductive_age = 5
    # reindeer_reproduction_rate = 0.5
    # reindeer_reproduction_energy = 0.5
    # reindeer_reproduction_distance = 10.0
    # reindeer_offspring_energy = 0.5

    # # Reindeer behavior parameters
    # reindeer_protected_range = 2.0
    # reindeer_visual_range = 15.0
    # reindeer_alert_range = 5.0
    # reindeer_grazing_rate = 0.2

    # #############################
    # ## Predator parameters
    # #############################

    # # Initial Predator parameters
    # predator_initial_age = 0
    # predator_max_age = 20
    # predator_energy = 1.0
    # predator_energy_decay = 0.01
    # predator_cruise_speed = 0.4
    # predator_hunt_speed = 1.5
    # predator_energy_threshold = 0.6

    # # Predator reproduction parameters
    # predator_reproductive_age = 5
    # predator_reproduction_rate = 0.5
    # predator_reproduction_energy = 0.5
    # predator_offspring_energy = 0.5
    # predator_reproduction_distance = 5.0

    # # Predator behavior parameters
    # predator_visual_range = 25.0
    # predator_eating_range = 1.0
    # predator_energy_gain = 0.4

    # #############################
    # ## Intrusion parameters
    # #############################
    # intrusion_center = None  # Middle of the top edge
    # intrusion_radius = None  # Radius of the exclusion zone

    # Load the configuration
    config = load_config("./config_initial.json")

    # Assign variables for easy access
    simulation = config["simulation"]
    reindeer = config["reindeer"]
    predator = config["predator"]
    intrusion = config["intrusion"]

    # Assign individual variables for simulation parameters
    animation = simulation["animation"]
    grid_size = tuple(simulation["grid_size"])
    num_reindeer = simulation["num_reindeer"]
    num_predators = simulation["num_predators"]
    max_steps = simulation["max_steps"]
    food_regeneration_rate = simulation["food_regeneration_rate"]
    reproduction_interval = simulation["reproduction_interval"]
    culling_rate = simulation["culling_rate"]
    culling_threshold = simulation["culling_threshold"]
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
    reindeer_reproduction_distance = reindeer["reproduction"]["reproduction_distance"]
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
    predator_reproduction_distance = predator["reproduction"]["reproduction_distance"]

    # Assign individual variables for predator behavior parameters
    predator_visual_range = predator["behavior"]["visual_range"]
    predator_eating_range = predator["behavior"]["eating_range"]
    predator_energy_gain = predator["behavior"]["energy_gain"]

    #############################
    ## Intrusion parameters
    #############################

    # Assign individual variables for intrusion parameters
    intrusion_center = intrusion["center"]
    intrusion_radius = intrusion["radius"]

    #############################
    ## End of configuration
    #############################

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

    # Data storage
    reindeer_population = []
    predator_population = []
    culling_statistics = []
    death_by_culling = [[0,0]]
    death_by_age = [[0,0]]
    death_by_predator= [[0,0]]
    death_by_starvation = [[0,0]]

    # Simulation loop
    for step in range(max_steps):
        if stop_loop:  # Check the flag
            print("Loop exited.")
            break

        # Update food grid (simple logistic regeneration)
        food_grid += food_regeneration_rate * (1 - food_grid)
        food_grid = np.clip(food_grid, 0, 1)  # Ensure food values stay within [0, 1]

        # Move agents
        temp_death_by_starvation=death_by_starvation[-1][1]
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
        death_by_starvation.append([step, temp_death_by_starvation])

        temp_death_by_predator=death_by_predator[-1][1]
        temp_number_of_reindeers = len(reindeers)
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
        death_by_predator.append([step, temp_death_by_predator+temp_number_of_reindeers-len(reindeers)])

        # Save population counts
        reindeer_population.append(len(reindeers))
        predator_population.append(len(predators))

        # Reproduction every reproduction_interval steps
        if step % reproduction_interval == 0:
            temp_death_by_age = death_by_age[-1][1]
            for reindeer in reindeers[:]:
                reindeer.update_age()
                if reindeer.age>reindeer.max_age:
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
            death_by_age.append([step, temp_death_by_age])

            for predator in predators[:]:
                predator.update_age()
                if np.random.rand() < predator.reproduction_rate:
                    predator.reproduce(
                        predators=predators,
                        offspring_energy=predator_offspring_energy,
                        reproduction_distance=predator_reproduction_distance,
                        exclusion_center=intrusion_center,
                        exclusion_radius=intrusion_radius,
                    )

            if len(predators) == 0:
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
                num_to_remove = int(len(reindeers)*culling_rate)
                if len(reindeers) - num_to_remove < culling_threshold:
                    num_to_remove = len(reindeers) - culling_threshold
            else:
                num_to_remove = 0
            # Randomly select indices to remove using numpy
            indices_to_remove = np.random.choice(len(reindeers), num_to_remove, replace=False)
            # Remove the selected objects from the list
            objects_to_remove = [reindeers[i] for i in indices_to_remove]
            reindeers = [obj for i, obj in enumerate(reindeers) if i not in indices_to_remove]
            death_by_culling.append([step, death_by_culling[-1][1]+num_to_remove])
            culling_statistics.append([step, num_to_remove])

        # Intrusion logic
        if step == max_steps // 2:
            intrusion_center = (0, grid_size[1] // 2)
            intrusion_radius = 40.0

        # Plot the environment
        if animation == True:
            plt.imshow(
                food_grid,
                cmap='Greens',
                extent=(0, grid_size[1], 0, grid_size[0]),
            )
            if intrusion_center is not None and intrusion_radius is not None:
                circle = Circle((intrusion_center[1],intrusion_center[0]), intrusion_radius, color='grey', alpha=1)
                plt.gca().add_artist(circle)
            if reindeers:
                reindeer_positions = np.array([r.position for r in reindeers])
                plt.scatter(
                    reindeer_positions[:, 1],
                    reindeer_positions[:, 0],
                    c="blue",
                    label="Reindeer",
                    alpha=0.7,
                )
            if predators:
                predator_positions = np.array([p.position for p in predators])
                plt.scatter(
                    predator_positions[:, 1],
                    predator_positions[:, 0],
                    c="red",
                    label="Predators",
                    alpha=0.7,
                )
            plt.title(f"Step {step}")
            plt.legend()
            plt.pause(0.1)
            plt.clf()

        # Stop if no reindeer are left
        if len(reindeers) == 0:
            print("All reindeer have been hunted.")
            break
    print("--- %s seconds ---" % (time.time() - start_time))
    if stop_loop == False:
        plt.show()
    print("Simulation finished.")

    # Plot population dynamics
    plt.figure()
    plt.plot(reindeer_population, label="Reindeer Population", color="blue")
    plt.plot(predator_population, label="Predator Population", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Population Dynamics")
    plt.legend()
    plt.show()

    death_by_age=np.array(death_by_age)
    death_by_starvation=np.array(death_by_starvation)
    death_by_predator=np.array(death_by_predator)
    death_by_culling=np.array(death_by_culling)
    plt.plot(death_by_age[:, 0], death_by_age[:, 1], color="blue", label="Old age")
    plt.plot(death_by_starvation[:, 0], death_by_starvation[:, 1], color="green", label="Starved")
    plt.plot(death_by_predator[:, 0], death_by_predator[:, 1], color="red", label="Eaten")
    plt.plot(death_by_culling[:, 0], death_by_culling[:, 1], color="orange", label="Culled")
    plt.title("Cause of death")
    plt.xlabel("Time Step")
    plt.ylabel("Total amount")
    plt.legend()
    plt.show()

    culling_statistics=np.array(culling_statistics)
    plt.plot(culling_statistics[:,0], culling_statistics[:,1])
    plt.title("Culling statistics")
    plt.xlabel("Time Step")
    plt.ylabel("Amount culled each season")
    plt.show()
if __name__ == "__main__":
    main()
