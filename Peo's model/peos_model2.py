#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

# Helper functions
def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def reflective_boundaries(position, velocity, grid_size, exclusion_center, exclusion_radius):
    """
    Reflects position and inverts velocity if an agent hits grid boundaries or enters an exclusion zone.
    Uses integer operations where applicable for better performance.
    """
    height, width = grid_size

    # Reflect against rectangular boundaries (with integer clamping)
    position[0] = min(max(position[0], 0), height)
    position[1] = min(max(position[1], 0), width)

    if position[0] == 0 or position[0] == height:
        velocity[0] = -velocity[0]
    if position[1] == 0 or position[1] == width:
        velocity[1] = -velocity[1]

    # Skip exclusion zone logic for performance unless explicitly needed
    if exclusion_center is not None and exclusion_radius is not None:
        dist_to_exclusion_center = np.linalg.norm(position - exclusion_center)
        if dist_to_exclusion_center < exclusion_radius:
            normal = (position - exclusion_center) / (dist_to_exclusion_center + 1e-5)
            velocity -= 2 * np.dot(velocity, normal) * normal
            position = exclusion_center + normal * exclusion_radius

    return position, velocity

class Reindeer:
    def __init__(self, x, y, age=0, max_speed=1.0, grazing_speed=0.2, energy_decay=0.02):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.energy = 1.0
        self.age = age
        self.max_speed = max_speed  # Speed limit for reindeer
        self.grazing_speed = grazing_speed  # Speed for grazing
        self.reproductive_age = 5
        self.max_age = 15
        self.reproduction_rate = 0.5
        self.energy_to_reproduce = 0.5
        self.energy_decay = energy_decay
        self.fleeing = False

    def move(self, predators, food_grid, reindeers, protected_range=1.0, visual_range=10.0, predatory_range=10.0, exclusion_center=None, exclusion_radius=None):
        # Initialize forces
        cohesion_force = np.array([0.0, 0.0])
        alignment_force = np.array([0.0, 0.0])
        separation_force = np.array([0.0, 0.0])
        food_force = np.array([0.0, 0.0])
        predator_force = np.array([0.0, 0.0])
        neighbors = 0

        # Boid behavior: cohesion, alignment, and separation
        for other in reindeers:
            if other is self:
                continue

            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            squared_distance = dx**2 + dy**2

            # Separation: Avoid collisions with nearby reindeer
            if squared_distance < protected_range**2:
                separation_force += np.array([dx, dy]) / (squared_distance + 1e-5)

            # Cohesion and alignment: Flocking behavior
            elif squared_distance < visual_range**2:
                cohesion_force += other.position
                alignment_force += other.velocity
                neighbors += 1

        # Normalize boid forces if there are neighbors
        if neighbors > 0:
            cohesion_force = (cohesion_force / neighbors - self.position) * 0.015
            alignment_force = (alignment_force / neighbors - self.velocity) * 0.015
            separation_force *= 0.1
        
        if len(predators) > 0:
            for predator in predators:
                dx = self.position[0] - predator.position[0]
                dy = self.position[1] - predator.position[1]
                predator_dist = np.sqrt(dx**2 + dy**2)

                if predator_dist < predatory_range:
                    self.fleeing = True
                    predator_force += np.array([dx, dy]) / (predator_dist + 1e-5)
                else:
                    self.fleeing = False
        
        predator_force *= 0.5  # Amplify predator force to ensure it dominates in flight scenarios

        # # Food-seeking behavior (ignored if fleeing)
        # if not self.fleeing:
        #     x_pos, y_pos = np.round(self.position).astype(int)
        #     x_pos = min(max(x_pos, 0), food_grid.shape[0] - 1)
        #     y_pos = min(max(y_pos, 0), food_grid.shape[1] - 1)
        #     x_pos = food_grid.shape[0] - 1 - x_pos

        #     food_x, food_y = np.unravel_index(np.argmax(food_grid), food_grid.shape)
        #     food_x = food_grid.shape[0] - 1 - food_x  # Invert x-axis for correct simulation coordinates
        #     food_position = np.array([food_x, food_y])
        #     food_dist = np.linalg.norm(food_position - np.array([self.position[0], self.position[1]]))

        #     if food_dist < visual_range:  # Prioritize food if within range
        #         food_force = (food_position - np.array([self.position[0], self.position[1]])) / (food_dist + 1e-5)
        #     food_force *= 1.0

        if not self.fleeing:
            # Runda positionen till närmaste matrisindex
            x_pos, y_pos = np.round(self.position).astype(int)
            x_pos = min(max(x_pos, 0), food_grid.shape[0] - 1)
            y_pos = min(max(y_pos, 0), food_grid.shape[1] - 1)
            x_pos = food_grid.shape[0] - 1 - x_pos  # Spegelvänd x för korrekt grid-referens

            # Definiera kvadraten att söka i, konvertera till int för slicing
            visual_range_half = int(visual_range // 2)
            x_min = max(x_pos - visual_range_half, 0)
            x_max = min(x_pos + visual_range_half + 1, food_grid.shape[0])
            y_min = max(y_pos - visual_range_half, 0)
            y_max = min(y_pos + visual_range_half + 1, food_grid.shape[1])

            # Extrahera sub-griden för mat
            sub_grid = food_grid[x_min:x_max, y_min:y_max]

            # Hitta index för mest mat i sub-griden
            local_max_index = np.unravel_index(np.argmax(sub_grid), sub_grid.shape)
            food_x = x_min + local_max_index[0]
            food_y = y_min + local_max_index[1]

            # Beräkna riktningen till den valda matcellen
            target_position = np.array([food_grid.shape[0] - 1 - food_x, food_y])  # Spegelvänd x tillbaka
            food_force = (target_position - self.position) / (np.linalg.norm(target_position - self.position) + 1e-5)
            food_force *= 0.01

        
        # Combine all forces
        total_force = 0.5*(cohesion_force + alignment_force + separation_force + food_force + predator_force)
    
        # Update velocity
        self.velocity += total_force
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        elif speed < self.grazing_speed:
            self.velocity = self.velocity / speed * self.grazing_speed

        # Update position
        self.position += self.velocity
        self.position, self.velocity = reflective_boundaries(self.position, self.velocity, grid_size, exclusion_center, exclusion_radius)
        self.energy -= self.energy_decay

    def graze(self, food_grid, grazing_rate=0.2):
        """
        Grazing behavior: Consume food from the grid to regain energy.

        :param food_grid: A 2D numpy array representing the food availability on the grid.
        :param grazing_rate: The maximum amount of food consumed per time step while grazing.
        """
        if not self.fleeing:  # Grazing occurs only if the reindeer is not fleeing
            # Convert position to integer grid coordinates for matrix indexing
            x, y = np.round(self.position).astype(int)
            x = min(max(x, 0), food_grid.shape[0] - 1)
            y = min(max(y, 0), food_grid.shape[1] - 1)
            x = food_grid.shape[0] - 1 - x  # Invert x-axis for correct indexing
            # Grazing on the current grid cell
            consumed_food = min(food_grid[x, y], grazing_rate)
            food_grid[x, y] -= consumed_food
            food_grid[x, y] = max(food_grid[x, y], 0.0)  # Food cannot be negative
            self.energy += 0.2*consumed_food
            self.energy = min(self.energy, 1.0)  # Cap energy at 1.0

    def reproduce(self, reindeers, grid_size, reproduction_distance=5.0, offspring_energy=0.5, exclusion_center=None, exclusion_radius=None):
        """
        Simulates reproduction for the reindeer.

        :param reindeers: List of all reindeer in the simulation.
        :param grid_size: Tuple representing the dimensions of the grid.
        :param reproduction_distance: Maximum distance offspring can spawn from the parent.
        :param offspring_energy: Energy level assigned to offspring.
        :param exclusion_center: Center of exclusion zone, if any.
        :param exclusion_radius: Radius of exclusion zone, if any.
        """
        if self.age >= self.reproductive_age and self.energy >= self.energy_to_reproduce:
            # Generate offspring position near parent within reproduction_distance
            offset = np.random.uniform(-reproduction_distance, reproduction_distance, size=2)
            offspring_position = self.position + offset

            # Reflect position within grid boundaries and adjust velocity if needed
            velocity_placeholder = np.array([0.0, 0.0])  # Offspring starts stationary
            offspring_position, _ = reflective_boundaries(offspring_position, velocity_placeholder, grid_size, exclusion_center, exclusion_radius)

            # Create offspring with slightly randomized properties
            offspring = Reindeer(
                x=offspring_position[0],
                y=offspring_position[1],
                age=0,
                max_speed=self.max_speed,
                grazing_speed=self.grazing_speed,
                energy_decay=self.energy_decay
            )
            offspring.energy = offspring_energy

            # Add offspring to the simulation
            reindeers.append(offspring)

            # Deduct energy from parent
            self.energy -= 0.5 * self.energy_to_reproduce

    def update_age(self):
        self.age += 1
        if self.age > self.max_age:
            self.energy = 0.0

class Predator:
    def __init__(self, x, y, age=0, cruise_speed=0.4, hunt_speed=1.5, energy=1.0, energy_threshold=0.6, energy_decay=0.01):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.cruise_speed = cruise_speed   # Speed for patrolling
        self.hunt_speed = hunt_speed       # Speed for hunting
        self.energy = energy               # Energy level
        self.energy_threshold = energy_threshold  # Energy threshold for hunting
        self.energy_decay = energy_decay   # Energy decay rate
        self.age = age
        self.reproductive_age = 2
        self.max_age = 15
        self.reproduction_rate = 0.25
        self.reproduction_energy = 0.4

    def move(self, prey, grid_size, visual_range=20.0, std_dev=0.5, exclusion_center=None, exclusion_radius=None):
        """
        Flytta predatorn baserat på energi och närvaro av byte.
        """
        # Predatorns tillstånd: Jakt eller patrullering
        direction = self.velocity / np.linalg.norm(self.velocity)
        if self.energy < self.energy_threshold:
            # Hitta närmaste byte att jaga
            target = None
            min_distance = float('inf')

            for p in prey:
                dist = distance(self.position, p.position)
                if dist < visual_range and dist < min_distance:
                    min_distance = dist
                    target = p

            if target is not None:
                # Jaga närmaste byte
                direction = target.position - self.position
                self.velocity = (direction / np.linalg.norm(direction)) * self.hunt_speed
            else:
                # No prey in sight, continue patrolling
                #direction += np.random.normal(0, std_dev, 2)  # Gaussian adjustment
                self.velocity = (direction / np.linalg.norm(direction)) * self.hunt_speed
        else:
            # Patrolling: Adjust direction with Gaussian noise
            #direction += np.random.normal(0, std_dev, 2)  # Gaussian adjustment
            self.velocity = (direction / np.linalg.norm(direction)) * self.cruise_speed

        # Update position
        self.position += self.velocity
        # Reflect at boundaries
        self.position, self.velocity = reflective_boundaries(self.position, self.velocity, grid_size, exclusion_center, exclusion_radius)
        # Reduce energy over time
        self.energy -= self.energy_decay

    def eat(self, prey, eating_range=1.0, energy_gain=0.4):
        """
        Ät upp ett byte om det är inom räckhåll.
        """
        for p in prey:
            dist = distance(self.position, p.position)
            if dist < eating_range:
                prey.remove(p)  # Ta bort bytet från listan
                self.energy += energy_gain  # Få energi från att äta
                self.energy = min(self.energy, 1.0)  # Energin får inte överstiga max
                break

    def reproduce(self, predators, grid_size, reproduction_distance=5.0, offspring_energy=0.5, exclusion_center=None, exclusion_radius=None):
        """
        Simulates reproduction for the predator.

        :param predators: List of all predators in the simulation.
        :param grid_size: Tuple representing the dimensions of the grid.
        :param reproduction_distance: Maximum distance offspring can spawn from the parent.
        :param offspring_energy: Energy level assigned to offspring.
        :param exclusion_center: Center of exclusion zone, if any.
        :param exclusion_radius: Radius of exclusion zone, if any.
        """
        # Check reproduction conditions
        if self.energy >= self.reproduction_energy and self.age >= self.reproductive_age:  # Use energy threshold as a condition for reproduction
            # Generate offspring position near parent within reproduction_distance
            offset = np.random.uniform(-reproduction_distance, reproduction_distance, size=2)
            offspring_position = self.position + offset

            # Reflect position within grid boundaries
            velocity_placeholder = np.array([0.0, 0.0])  # Offspring starts stationary
            offspring_position, _ = reflective_boundaries(offspring_position, velocity_placeholder, grid_size, exclusion_center, exclusion_radius)

            # Create offspring with similar properties
            offspring = Predator(
                x=offspring_position[0],
                y=offspring_position[1],
                cruise_speed=self.cruise_speed,
                hunt_speed=self.hunt_speed,
                energy=offspring_energy,
                energy_threshold=self.energy_threshold,
                energy_decay=self.energy_decay
            )

            # Add offspring to the simulation
            predators.append(offspring)

            # Deduct energy from parent
            self.energy -= 0.5 * self.energy_threshold

    def update_age(self):
        self.age += 1
        if self.age > self.max_age:
            self.energy = 0.0

# %%

# Simulation parameters
grid_size = (200, 300)  # Height, Width
num_reindeer = 200  # Increased number of reindeer
num_predators = 5
max_steps = 10000
food_regeneration_rate = 0.003  # Slightly faster food regeneration
reproduction_interval = 35  # Reproduction occurs less frequently

# Initialize food grid
food_grid = np.random.uniform(0.2, 0.8, grid_size)


# Test a food grid with a single peak
# food_grid = np.zeros(grid_size)
# food_grid[20:40, 100:120] = np.random.uniform(0.5, 1.0, (20, 20))
# food_grid[80:100, 100:120] = np.random.uniform(0.5, 1.0, (20, 20))
# food_grid[20:40, 150:170] = np.random.uniform(0.5, 1.0, (20, 20))

# Create a custom colormap
from matplotlib.colors import ListedColormap, BoundaryNorm
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
        np.random.uniform(0, grid_size[0]), 
        np.random.uniform(0, grid_size[1]), 
        age=np.random.randint(0, 15)  # Random age between 0 and 15
    ) 
    for _ in range(num_reindeer)
]

predators = [
    Predator(
        np.random.uniform(0, grid_size[0]), 
        np.random.uniform(0, grid_size[1]),
        age=np.random.randint(0, 15),  # Random age between 0 and 15 
        energy_decay=0.01,
    ) 
    for _ in range(num_predators)
]

# Data storage
reindeer_population = []
predator_population = []

# Simulation loop
for step in range(max_steps):
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