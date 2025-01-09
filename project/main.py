import numpy as np
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

# from matplotlib.colors import ListedColormap, BoundaryNorm
from helper import *
from animate import *
from simulation import *
from pynput import keyboard
import os

stop_loop = False

start_time = time.time()
CONFIG_PATH = "./config.json"
RESULT_PATH = "./results/"
IS_BATCH = True


# Function to handle key press events
def on_press(key):
    global stop_loop
    if key == keyboard.Key.esc:  # If Escape is pressed
        print("Escape key pressed! Stopping the loop...")
        stop_loop = True
        return False  # Stops the listener


# Initialize the simulation
def run_simulation(
    seed_number, config, isAnimate, capture_interval, max_steps, isPlotResults
):
    simulation = Simulation(
        config=config,
        seed_number=seed_number,
        isAnimate=isAnimate,
        capture_interval=capture_interval,
        result_path=RESULT_PATH,
        is_batch=IS_BATCH,
        isPlotResults=isPlotResults,
        max_steps=max_steps,
    )
    simulation.simulate()
    simulation.save_result()
    current_time = time.strftime("%Y%m%d-%H%M%S")
    print("Finish time: ", current_time)
    print(f"Simulation with seed {seed_number} is done.")


def main():
    # Load the configuration
    config = load_config(CONFIG_PATH)

    # Assign variables for easy access
    simulation = config["simulation"]

    # Assign individual variables for simulation parameters
    isAnimate = simulation["is_animate"]
    isPlotResults = simulation["is_plot_results"]
    capture_interval = simulation["capture_interval"]
    seed_list = simulation["seed_list"]
    max_steps = simulation["max_steps"]
    is_batch = simulation["is_batch"]

    if is_batch:
        IS_BATCH = True
    else:
        IS_BATCH = False

    #############################
    ## End of configuration
    #############################

    ############################
    ## Warning parameters
    ############################
    if IS_BATCH:
        if isAnimate:
            print("Warning: Animation is disabled in batch mode.")
            isAnimate = False
        if isPlotResults:
            print("Warning: Plotting results is disabled in batch mode.")
            isPlotResults = False
        if capture_interval <= 10000:
            capture_interval = 999999999
            print("Warning: Batch will not capture image")

            # seed_number = simulation["seed_number"]
        if len(seed_list) == 0:
            print("Warning: Seed list is empty. Using a random seed.")
            seed_list.append(np.random.seed())
        print("Running with Batch mode")

    num_parallel = simulation["num_parallel"]
    if num_parallel == 0:
        num_cores = os.cpu_count()
        num_parallel = max(1, int(num_cores * 0.4))
        print(
            f"No configuration for number of cores. Using 60% of available cores: {num_parallel}"
        )
    if IS_BATCH:
        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            list(
                executor.map(
                    run_simulation,
                    seed_list,
                    [config] * len(seed_list),
                    [isAnimate] * len(seed_list),
                    [capture_interval] * len(seed_list),
                    [max_steps] * len(seed_list),
                    [isPlotResults] * len(seed_list),
                )
            )
    else:
        run_simulation(
            seed_list[0], config, isAnimate, capture_interval, max_steps, isPlotResults
        )


if __name__ == "__main__":
    start_time_batch = time.time()
    parser = argparse.ArgumentParser(description="Run script with a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the config file",
        default="./config.json",
    )
    parser.add_argument(
        "--result",
        type=str,
        required=False,
        help="Path to store result file",
        default="./results/",
    )
    parser.add_argument(
        "--batch",
        type=str,
        required=False,
        help="Path to store result file",
        default="False",
    )
    args = parser.parse_args()
    CONFIG_PATH = args.config
    RESULT_PATH = args.result
    is_batch_text = args.batch

    if is_batch_text == "False":
        IS_BATCH = False
    else:
        IS_BATCH = True

    if not IS_BATCH:
        print("Batch mode is disabled.")
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    else:
        print("Batch mode is enabled.")

    print(f"Config path: {CONFIG_PATH}")
    print(f"Result path: {RESULT_PATH}")
    print("Starting simulation...")
    main()

    end_time_batch = time.time()
    diff_time = end_time_batch - start_time_batch
    print(
        f"Total time: {diff_time} seconds: or {diff_time/60} minutes : {diff_time/3600} hours"
    )
