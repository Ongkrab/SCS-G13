from helper import reflective_boundaries, distance
import numpy as np

class Predator:
    def __init__(self, x, y, age, max_age, energy, energy_decay, reproductive_age, reproduction_rate, reproduction_energy, cruise_speed, hunt_speed, energy_threshold):
        
        """
        Initializes a predator object with the specified properties.
        Example:
            age = 0,
            cruise_speed = 0.4,
            hunt_speed = 1.5,
            energy = 1.0,
            energy_threshold = 0.6,
            energy_decay = 0.01,
            grid_size = (200,
            300 ) 
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.cruise_speed = cruise_speed   # Speed for patrolling
        self.hunt_speed = hunt_speed       # Speed for hunting
        self.energy = energy               # Energy level
        self.energy_threshold = energy_threshold  # Energy threshold for hunting
        self.energy_decay = energy_decay   # Energy decay rate
        self.age = age
        self.max_age = max_age                   # Maximum age
        self.reproductive_age = reproductive_age          # Age when reproduction is possible
        self.reproduction_rate = reproduction_rate       # Probability of reproducing
        self.reproduction_energy = reproduction_energy      # Energy threshold for reproduction


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
        self.position, self.velocity = reflective_boundaries(self.position, self.velocity, exclusion_center, exclusion_radius)
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
            offspring_position, _ = reflective_boundaries(offspring_position, velocity_placeholder, exclusion_center, exclusion_radius)

            # Create offspring with similar properties
            offspring = Predator(
                x=offspring_position[0],
                y=offspring_position[1],
                age = 0,
                max_age = self.max_age,
                energy = self.energy,
                energy_decay = self.energy_decay,
                reproductive_age = self.reproductive_age,
                reproduction_rate = self.reproduction_rate,
                reproduction_energy = self.reproduction_energy,
                cruise_speed = self.cruise_speed,
                hunt_speed = self.hunt_speed,
                energy_threshold = self.energy_threshold
            )

            # Add offspring to the simulation
            predators.append(offspring)

            # Deduct energy from parent
            self.energy -= 0.5 * self.reproduction_energy

    def update_age(self):
        self.age += 1
        if self.age > self.max_age:
            self.energy = 0.0
