import numpy as np
from helper import reflective_boundaries, distance


class Reindeer:
    def __init__(
        self,
        x,
        y,
        age,
        max_age,
        energy,
        energy_decay,
        reproductive_age,
        reproduction_rate,
        reproduction_energy,
        energy_threshold,
        max_speed,
        grazing_speed,
    ):
        """
        Initializes a reindeer object with the specified properties.
        example:
            age=0,
            max_speed=1.0,
            grazing_speed=0.2,
            energy_decay=0.02,
            self.reproductive_age = 5
            self.max_age = 15
            self.reproduction_rate = 0.5
            self.energy_to_reproduce = 0.5
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.energy = energy
        self.age = age
        self.max_speed = max_speed  # Speed limit for reindeer
        self.grazing_speed = grazing_speed  # Speed for grazing
        self.reproductive_age = reproductive_age
        self.max_age = max_age
        self.reproduction_rate = reproduction_rate
        self.reproduction_energy = reproduction_energy
        self.energy_threshold = (
            energy_threshold  # Energy threshold for increased food search
        )
        # self.reproductive_age = 5
        # self.max_age = 15
        # self.reproduction_rate = 0.5
        # self.energy_to_reproduce = 0.5
        self.energy_decay = energy_decay
        self.fleeing = False

    def get_position(self):
        return self.position

    def move(
        self,
        food_grid,
        predators,
        reindeers,
        protected_range,
        visual_range,
        alert_range,
        exclusion_center=None,
        exclusion_radius=None,
    ):
        """
        Simulates the movement of the reindeer based on various forces.
        food_grid: A 2D numpy array representing the food availability on the grid.
        predators: List of all predators in the simulation.
        reindeers: List of all reindeer in the simulation.
        protected_range: Minimum distance to maintain between reindeer.
        visual_range: Maximum distance at which a reindeer can detect others.
        alert_range: Distance at which a predator triggers the fleeing response.
        exclusion_center: Center of exclusion zone, if any.
        exclusion_radius: Radius of exclusion zone, if any.
        """
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

                if predator_dist < alert_range:
                    self.fleeing = True
                    predator_force += np.array([dx, dy]) / (predator_dist + 1e-5)
                else:
                    self.fleeing = False

        predator_force *= (
            0.5  # Amplify predator force to ensure it dominates in flight scenarios
        )

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
            x_pos = (
                food_grid.shape[0] - 1 - x_pos
            )  # Spegelvänd x för korrekt grid-referens

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
            target_position = np.array(
                [food_grid.shape[0] - 1 - food_x, food_y]
            )  # Spegelvänd x tillbaka
            food_force = (target_position - self.position) / (
                np.linalg.norm(target_position - self.position) + 1e-5
            )
            food_force *= 0.01
            if self.energy < self.energy_threshold:
                food_force *= 2

        # Combine all forces
        total_force = 0.5 * (
            cohesion_force
            + alignment_force
            + separation_force
            + food_force
            + predator_force
        )

        # Update velocity
        self.velocity += total_force
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        elif speed < self.grazing_speed:
            self.velocity = self.velocity / speed * self.grazing_speed

        # Update position
        self.position += self.velocity
        self.position, self.velocity = reflective_boundaries(
            self.position, self.velocity, exclusion_center, exclusion_radius
        )
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
            self.energy += 0.2 * consumed_food
            self.energy = min(self.energy, 1.0)  # Cap energy at 1.0

    def reproduce(
        self,
        reindeers,
        reproduction_distance,
        offspring_energy,
        exclusion_center=None,
        exclusion_radius=None,
    ):
        """
        Simulates reproduction for the reindeer.

        :param reindeers: List of all reindeer in the simulation.
        :param grid_size: Tuple representing the dimensions of the grid.
        :param reproduction_distance: Maximum distance offspring can spawn from the parent.
        :param offspring_energy: Energy level assigned to offspring.
        :param exclusion_center: Center of exclusion zone, if any.
        :param exclusion_radius: Radius of exclusion zone, if any.
        """
        if (
            self.age >= self.reproductive_age
            and self.energy >= self.reproduction_energy
        ):
            # Generate offspring position near parent within reproduction_distance
            offset = np.random.uniform(
                -reproduction_distance, reproduction_distance, size=2
            )
            offspring_position = self.position + offset

            # Reflect position within grid boundaries and adjust velocity if needed
            velocity_placeholder = np.array([0.0, 0.0])  # Offspring starts stationary
            offspring_position, _ = reflective_boundaries(
                offspring_position,
                velocity_placeholder,
                exclusion_center,
                exclusion_radius,
            )

            # Create offspring with slightly randomized properties
            offspring = Reindeer(
                x=offspring_position[0],
                y=offspring_position[1],
                age=0,
                max_age=self.max_age,
                energy=offspring_energy,
                energy_decay=self.energy_decay,
                reproductive_age=self.reproductive_age,
                reproduction_rate=self.reproduction_rate,
                reproduction_energy=self.reproduction_energy,
                max_speed=self.max_speed,
                grazing_speed=self.grazing_speed,
                energy_threshold=self.energy_threshold,
            )

            # Add offspring to the simulation
            reindeers.append(offspring)

            # Deduct energy from parent
            self.energy -= 0.5 * self.reproduction_energy

    def update_age(self):
        self.age += 1
        # if self.age > self.max_age:
        #    self.energy = 0.0
