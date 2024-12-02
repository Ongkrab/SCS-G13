import numpy as np
import keyboard
import matplotlib.pyplot as plt
from predator import Predator
from reindeer import Reindeer
from matplotlib.colors import ListedColormap, BoundaryNorm
from helper import *
from pynput import keyboard

def main():  
    # Simulation parameters
    grid_size = (200, 300)  # Height, Width
    num_reindeer = 200  # Increased number of reindeer
    num_predators = 5
    max_steps = 10000
    food_regeneration_rate = 0.003  # Slightly faster food regeneration
    reproduction_interval = 35  # Reproduction occurs less frequently

    # Reindeer parameters
    reindeer_initial_age=0
    reindeer_max_speed=1.0
    reindeer_max_age = 15
    reindeer_energy = 1.0
    reindeer_grazing_speed=0.2
    reindeer_energy_decay=0.02
    reindeer_reproductive_age = 5
    reindeer_reproduction_rate = 0.5
    reindeer_reproduction_energy = 0.5

    # Predator parameters
    predator_initial_age=0
    predator_max_age = 20
    predator_energy = 1.0
    predator_energy_decay = 0.01
    predator_reproductive_age = 5
    predator_reproduction_rate = 0.5
    predator_reproduction_energy = 0.5
    predator_cruise_speed = 0.4
    predator_hunt_speed = 1.5
    predator_energy_threshold = 0.6

    # Initialize food grid
    food_grid = np.random.uniform(0.2, 0.8, grid_size)

    # Test a food grid with a single peak
    # food_grid = np.zeros(grid_size)
    # food_grid[20:40, 100:120] = np.random.uniform(0.5, 1.0, (20, 20))
    # food_grid[80:100, 100:120] = np.random.uniform(0.5, 1.0, (20, 20))
    # food_grid[20:40, 150:170] = np.random.uniform(0.5, 1.0, (20, 20))

    # Create a custom colormap
    cmap = plt.cm.Greens  # Use Greens for positive values
    colors = cmap(np.linspace(0, 1, 256))  # Get the original colors
    colors[0] = np.array([0.5, 0.5, 0.5, 1])  # Replace the first color with gray (RGBA)
    custom_cmap = ListedColormap(colors)
    # Normalize values
    norm = BoundaryNorm([-0.1, 0, 1], ncolors=custom_cmap.N, clip=True)

    # Intrusion parameters, None at start
    intrusion_center = None  # Middle of the top edge
    intrusion_radius = None  # Radius of the exclusion zone


    # Initialize agents with random ages
    reindeers = [
        Reindeer(
            x = np.random.uniform(0, grid_size[0]), 
            y =np.random.uniform(0, grid_size[1]), 
            age = np.random.randint(0, 15),  # Random age between 0 and 15
            max_age = reindeer_max_age,
            energy = reindeer_energy,
            energy_decay = reindeer_energy_decay,
            reproductive_age = reindeer_reproductive_age,
            reproduction_rate = reindeer_reproduction_rate,
            reproduction_energy = reindeer_reproduction_energy,
            max_speed = reindeer_max_speed,
            grazing_speed = reindeer_grazing_speed,
        ) 
        for _ in range(num_reindeer)
    ]

    predators = [
        Predator(
            x = np.random.uniform(0, grid_size[0]), 
            y = np.random.uniform(0, grid_size[1]),
            age = np.random.randint(0, 15),  # Random age between 0 and 15 
            max_age = predator_max_age,
            energy = predator_energy,
            energy_decay = predator_energy_decay,
            reproductive_age = predator_reproductive_age,
            reproduction_rate = predator_reproduction_rate,
            reproduction_energy = predator_reproduction_energy,
            cruise_speed = predator_cruise_speed,
            hunt_speed = predator_hunt_speed,
            energy_threshold = predator_energy_threshold
        ) 
        for _ in range(num_predators)
    ]

    # Data storage
    reindeer_population = []
    predator_population = []

    # Simulation loop
    for step in range(max_steps):
        # if keyboard.is_pressed("esc"):  # Check if Escape is pressed
        #     print("Escape pressed! Exiting loop.")
        #     break
        
        # Update food grid (simple logistic regeneration)
        food_grid += food_regeneration_rate * (1 - food_grid)
        food_grid = np.clip(food_grid, 0, 1)  # Ensure food values stay within [0, 1]

        # Move agents
        for reindeer in reindeers[:]:
            reindeer.move(predators, food_grid, reindeers, protected_range=2.0, visual_range=15.0, exclusion_center=intrusion_center, exclusion_radius=intrusion_radius)  # Increased visual range
            reindeer.graze(food_grid, grazing_rate=0.2)  # Faster grazing
            if reindeer.energy <= 0:
                reindeers.remove(reindeer)  # Remove dead reindeer

        for predator in predators[:]:
            predator.move(reindeers, grid_size, visual_range=25.0, exclusion_center=intrusion_center, exclusion_radius=intrusion_radius)  # Increased hunting range
            predator.eat(reindeers, energy_gain=0.3)  # Higher energy gain from eating
            if predator.energy <= 0:
                predators.remove(predator)  # Remove dead predators

        # Save population counts
        reindeer_population.append(len(reindeers))
        predator_population.append(len(predators))

        # Reproduction every reproduction_interval steps
        if step % reproduction_interval == 0:
            for reindeer in reindeers[:]:
                reindeer.update_age()
                if np.random.rand() < reindeer.reproduction_rate:
                    reindeer.reproduce(reindeers, grid_size, reproduction_distance=10.0, exclusion_center=intrusion_center, exclusion_radius=intrusion_radius)
                
            for predator in predators[:]:
                predator.update_age()
                if np.random.rand() < predator.reproduction_rate:
                    predator.reproduce(predators, grid_size, reproduction_distance=10.0, exclusion_center=intrusion_center, exclusion_radius=intrusion_radius)
            
            if len(predators) == 0:
                predators = [
                    Predator(
                        np.random.uniform(0, grid_size[0]), 
                        np.random.uniform(0, grid_size[1]),
                        age=np.random.randint(0, 15),  # Random age between 0 and 15 
                        energy_decay=0.01,
                    ) 
                    for _ in range(num_predators)
                ]

        if step % reproduction_interval == 0:
            # Culling of the herd
            reindeers = reindeers[:-20] # Remove the last 20 reindeer
        
        # Intrusion logic
        if step == max_steps // 2:
            intrusion_center = (0, grid_size[1] // 2)
            intrusion_radius = 40.0
        
        # Plot the environment
        if intrusion_center is not None and intrusion_radius is not None:
            # Skapa en mask för intrusionszonen
            y, x = np.ogrid[:grid_size[0], :grid_size[1]]
            mirrored_intrusion_center = (grid_size[0] - 1 - intrusion_center[0], intrusion_center[1])  # Invert x-axis for correct simulation coordinates
            intrusion_mask = (x - mirrored_intrusion_center[1])**2 + (y - mirrored_intrusion_center[0])**2 <= intrusion_radius**2
            food_grid_with_intrusion = food_grid.copy()
            food_grid_with_intrusion[intrusion_mask] = -0.1  # Mark intrusion area as dark gray in plot
        else:
            food_grid_with_intrusion = food_grid

        # Plot the environment
        plt.imshow(food_grid_with_intrusion, cmap=custom_cmap, extent=(0, grid_size[1], 0, grid_size[0]))
        if reindeers:
            reindeer_positions = np.array([r.position for r in reindeers])
            plt.scatter(reindeer_positions[:, 1], reindeer_positions[:, 0], c='blue', label="Reindeer", alpha=0.7)
        if predators:
            predator_positions = np.array([p.position for p in predators])
            plt.scatter(predator_positions[:, 1], predator_positions[:, 0], c='red', label="Predators", alpha=0.7)
        plt.title(f"Step {step}")
        plt.legend()
        plt.pause(0.1)
        plt.clf()

        # Stop if no reindeer are left
        if len(reindeers) == 0:
            print("All reindeer have been hunted.")
            break


    plt.show()
    print("Simulation finished.")

    # Plot population dynamics
    plt.figure()
    plt.plot(reindeer_population, label="Reindeer Population", color='blue')
    plt.plot(predator_population, label="Predator Population", color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Population Dynamics")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()