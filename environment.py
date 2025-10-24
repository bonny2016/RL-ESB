# environment.py
import numpy as np 
import math
import os
from config import (OPERATION_START_MIN, OPERATION_END_MIN, T_RANGE, BUS_LINES, DEPOT, INITIAL_NUM_BUSES,
                    STATE_DIM, COORDINATES, MAX_DISTANCE, W_DEADHEAD, W_UNUSED_PENALTY, 
                    W_REST_REWARD, W_UNAVAILABILITY, W_DEMAND_PENALTY, W_CHAIN, W_FINAL)

class BusSchedulingEnv: 
    """
    Bus Scheduling Environment with time dynamics and chaining bonus.

    This class defines the environment for the bus scheduling problem. It handles
    the simulation of the bus network, including the timetable, bus status,
    state representation, and reward calculation.

    Attributes:
        timetable (list): A list of all the bus trips (events) to be scheduled.
        num_events (int): The total number of events in the timetable.
        current_index (int): The index of the current event in the timetable.
        bus_status (dict): A dictionary tracking the status of each bus (location, availability, etc.).
        schedule (dict): A dictionary to store the schedule for each bus.
        observation_space_dim (int): The dimension of the observation space.
        action_space_dim (int): The dimension of the action space.
    """
    def __init__(self):
        """
        Initializes the BusSchedulingEnv.
        """
        self.timetable = self.generate_timetable()
        self.num_events = len(self.timetable)
        global MAX_EPISODE_STEPS
        MAX_EPISODE_STEPS = self.num_events
        
        self.current_index = 0
        # For each bus, track: location, next_available_time, used flag.
        self.bus_status = {bus_id: {"location": DEPOT, "next_available_time": OPERATION_START_MIN, "used": False} 
                           for bus_id in range(INITIAL_NUM_BUSES)}
        self.schedule = {bus_id: [] for bus_id in range(INITIAL_NUM_BUSES)}
        
        # Define state and action spaces
        # State: [current_time/T_RANGE, current_line/num_lines] + [availability for each bus]
        self.observation_space_dim = 2 + INITIAL_NUM_BUSES
        # Action space is the number of buses that can be assigned
        self.action_space_dim = INITIAL_NUM_BUSES

    def generate_timetable(self):
        """
        Generates a timetable of bus trips based on the bus line definitions in the config file.

        Returns:
            list: A sorted list of events (bus trips).
        """
        events = []
        for line_id, info in BUS_LINES.items():
            interval = info["interval"]
            t = OPERATION_START_MIN
            while t <= OPERATION_END_MIN:
                event = {
                    "time": t,
                    "line_id": line_id,
                    "terminal": info["terminal"],
                    "trip_time": info["trip_time"],
                    "rest_time": info["rest_time"]
                }
                events.append(event)
                t += interval
        events.sort(key=lambda e: (e["time"], e["line_id"]))
        return events

    def get_valid_actions(self):
        """
        Returns a list of valid bus IDs that can be assigned to the current event.
        A bus is valid if it will be available by the time of the current event.

        Returns:
            list: A list of valid bus IDs.
        """
        if self.current_index >= self.num_events:
            return []
            
        event = self.timetable[self.current_index]
        event_time = event["time"]
        
        valid_actions = []
        for bus_id in range(INITIAL_NUM_BUSES):
            if self.bus_status[bus_id]["next_available_time"] <= event_time:
                valid_actions.append(bus_id)
                
        return valid_actions

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        self.current_index = 0
        self.bus_status = {bus_id: {"location": DEPOT, "next_available_time": OPERATION_START_MIN, "used": False} 
                           for bus_id in range(INITIAL_NUM_BUSES)}
        self.schedule = {bus_id: [] for bus_id in range(INITIAL_NUM_BUSES)}
        return self.get_state()

    def get_state(self):
        """
        Builds the state vector for the current event.

        The state vector consists of:
        - Normalized current event time.
        - Normalized bus line ID.
        - Continuous availability value for each bus.

        Returns:
            np.ndarray: The state vector.
        """
        if self.current_index >= self.num_events:
            return np.zeros(STATE_DIM, dtype=np.float32)
        event = self.timetable[self.current_index]
        norm_time = event["time"] / OPERATION_END_MIN
        norm_line = event["line_id"] / len(BUS_LINES)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = norm_time
        state[1] = norm_line
        for i in range(INITIAL_NUM_BUSES):
            # Availability value: positive means available; negative means not ready.
            avail_value = (event["time"] - self.bus_status[i]["next_available_time"]) / T_RANGE
            state[2 + i] = avail_value
        return state

    def compute_deadhead_cost(self, current_location, required_location):
        """
        Computes the deadhead cost for moving a bus between two locations.

        Args:
            current_location (str): The current location of the bus.
            required_location (str): The required location of the bus.

        Returns:
            float: The normalized deadhead cost.
        """
        x1, y1 = COORDINATES[current_location]
        x2, y2 = COORDINATES[required_location]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        normalized_cost = distance / MAX_DISTANCE
        return normalized_cost

    def step(self, action):
        """
        Executes a single step in the environment.

        Args:
            action (int): The action to take (i.e., the bus to assign).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The next state.
                - float: The reward for the current step.
                - bool: Whether the episode has finished.
                - dict: A dictionary containing additional information.
        """
        if self.current_index >= self.num_events:
            return self.get_state(), 0, True, {}

        event = self.timetable[self.current_index]
        event_time = event["time"]
        required_terminal = event["terminal"]

        bus_info = self.bus_status[action]
        current_location = bus_info["location"]

        # --- Action Masking should already ensure that the selected bus is available,
        # but we also include a penalty for unavailability as safety.
        penalty_unavail = 0.0
        if bus_info["next_available_time"] > event_time:
            # Use configured penalty for buses not ready in time
            penalty_unavail = W_UNAVAILABILITY

        # Deadhead cost: if bus is not at required terminal.
        if current_location != required_terminal:
            deadhead_cost = self.compute_deadhead_cost(current_location, required_terminal)
        else:
            deadhead_cost = 0.0

        # Unused bus penalty (rn): if chosen bus is unused while another used bus is available at the depot.
        rn = 0.0
        if (not bus_info["used"]) and any(self.bus_status[b]["used"] and 
                                          self.bus_status[b]["location"] == DEPOT and 
                                          self.bus_status[b]["next_available_time"] <= event_time
                                          for b in self.bus_status):
            rn = 1.0

        # Chain bonus (W_CHAIN): if the chosen bus has been used before and its last event was on the same bus line.
        chain_bonus = 0.0
        if self.schedule[action]:
            last_event = self.schedule[action][-1]
            if last_event["line_id"] == event["line_id"]:
                chain_bonus = W_CHAIN

        # Rest reward (rk): if bus is used and available.
        rk = 1.0 if (bus_info["used"] and bus_info["next_available_time"] <= event_time) else 0.0
        ru = 0.0  # demand penalty (not implemented here)

        step_reward = - (W_UNUSED_PENALTY * rn + W_DEADHEAD * deadhead_cost + penalty_unavail) \
                      + W_REST_REWARD * rk + chain_bonus - W_DEMAND_PENALTY * ru

        # Record event and update bus status.
        self.schedule[action].append(event)
        self.bus_status[action]["used"] = True
        self.bus_status[action]["location"] = required_terminal
        self.bus_status[action]["next_available_time"] = event_time + event["trip_time"] + event["rest_time"]

        self.current_index += 1
        done = (self.current_index >= self.num_events)
        next_state = self.get_state()

        if done:
            num_used = sum(1 for b in self.bus_status if self.bus_status[b]["used"])
            final_penalty = -W_FINAL * num_used
            step_reward += final_penalty

        info = {"event": event, "deadhead_cost": deadhead_cost, "rn": rn, "penalty_unavail": penalty_unavail, "rk": rk, "chain_bonus": chain_bonus}
        return next_state, step_reward, done, info

    def get_total_buses_used(self):
        """
        Returns the total number of buses that were used in the schedule.

        Returns:
            int: The total number of buses used.
        """
        return sum(1 for bus_id in self.bus_status if self.bus_status[bus_id]["used"])

    def print_and_save(self, text, file):
        """
        Helper function to print text to the console and write it to a file.

        Args:
            text (str): The text to print and save.
            file (file object): The file to write to.
        """
        print(text, end="")  # Print to console
        file.write(text)     # Write to file

    def print_problem(self, file_path="data/results.txt"):
        """
        Prints the problem definition to the console and a file.

        Args:
            file_path (str, optional): The path to the file to save the problem definition to. Defaults to "data/results.txt".
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            self.print_and_save("=== Problem Definition ===\n", f)
            start_hr, start_min = divmod(OPERATION_START_MIN, 60)
            end_hr, end_min = divmod(OPERATION_END_MIN, 60)
            self.print_and_save(f"Operation Period: {start_hr:02d}:{start_min:02d} to {end_hr:02d}:{end_min:02d}\n\n", f)
            self.print_and_save("Bus Lines:\n", f)
            for line_id, info in BUS_LINES.items():
                self.print_and_save(f"  Bus Line {line_id}: {info['name']} - Loop at {info['terminal']}, "
                                    f"Interval: {info['interval']} min (Trip: {info['trip_time']} + Rest: {info['rest_time']})\n", f)
            self.print_and_save("\nTimetable (Departure Events):\n", f)
            for event in self.timetable:
                hr, minute = divmod(event["time"], 60)
                self.print_and_save(f"  {hr:02d}:{minute:02d} - Bus Line {event['line_id']} (Terminal: {event['terminal']})\n", f)
            self.print_and_save("==========================\n\n", f)

    def print_solution(self, file_path="data/results.txt"):
        """
        Prints the final bus schedules to the console and a file.

        Args:
            file_path (str, optional): The path to the file to save the solution to. Defaults to "data/results.txt".
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as f:  # Append to the same file
            self.print_and_save("=== Final Bus Schedules (Solution) ===\n", f)
            for bus_id, events in self.schedule.items():
                if events:
                    self.print_and_save(f"Bus {bus_id} schedule:\n", f)
                    for event in events:
                        hr, minute = divmod(event["time"], 60)
                        self.print_and_save(f"  {hr:02d}:{minute:02d} - Bus Line {event['line_id']} (Terminal: {event['terminal']})\n", f)
                else:
                    self.print_and_save(f"Bus {bus_id} was not used.\n", f)
            self.print_and_save("========================================\n", f)

