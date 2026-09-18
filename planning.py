# planning.py
"""
Multi-Agent Path Planning Module with A* and Q-learning-based Collision Avoidance

This module implements a hierarchical path planning system for Automated Guided Vehicles (AGVs)
in warehouse environments. It combines A* algorithm for optimal path finding with a rule-based
Q-learning mechanism for dynamic collision avoidance.

Architecture:
    - Upper layer: Task assignment algorithms (Enumeration/Nearest Neighbor/GA)
    - Lower layer: This module handles path execution and conflict resolution

Key Features:
    - Static environment path planning
    - Real-time collision avoidance using Q-learning rules
    - Task completion verification with temporal constraints
    - Support for loading/unloading time requirements

Author: Research Team
Version: 2.0
Date: 2026-06-28
"""

import numpy as np
import yaml
import time
import os
from astar_class import AStar

# ============================================================================
# Constant Definitions
# ============================================================================

CUSTOM_MAP_PRIORITY = [
    'custom_map.yaml',
    'map_small_traditional_aisle.yaml',
    'map_medium_traditional_aisle.yaml',
    'map_large_traditional_aisle.yaml'
]

_MAP_FILE = None
for _map_file in CUSTOM_MAP_PRIORITY:
    if os.path.exists(_map_file):
        _MAP_FILE = _map_file
        break
if _MAP_FILE is None:
    raise FileNotFoundError(
        "未找到地图文件！请先运行 step1_create_instance.py 生成地图。\n"
        "期望的文件: custom_map.yaml 或 map_{small/medium/large}_traditional_aisle.yaml"
    )

# Action space: up, left, down, right, stay
ACTION_SET = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]])

# Node positions for local state observation (5x5 grid relative positions)
OBSERVATION_NODES = np.array([
    [0, 2], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1],
    [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [4, 2]
])

# Obstacle node positions in local observation space
OBSTACLE_NODES = np.array([[1, 2], [2, 1], [2, 3], [3, 2]])

# Base停留 time for normal waypoints (steps)
BASE_STOP_TIME = 1

# Loading/unloading time at task points (pickup/dropoff)
LOAD_UNLOAD_TIME = 1

# Maximum planning horizon to prevent infinite loops
MAX_PLANNING_STEPS = 300


class Planning:
    """
    Multi-agent path planner with collision avoidance.

    This class implements a two-layer planning approach:
    1. Global path planning using A* algorithm
    2. Local collision avoidance using rule-based Q-learning

    Attributes:
        output_file: Path to save the planning results in YAML format
        col: Map width (number of columns)
        row: Map height (number of rows)
        agent_size: Number of AGVs in the system
        obs: List of obstacle coordinates [[x,y], ...]
        task_pos: Current task position index for each AGV
        pick_or_place: Task phase tracker (0=idle, >0=in task)
        task_list: Ordered task sequence for each AGV
        agent_pos: Complete path history for each AGV
        custom_tasks: Original task definitions
        state: Environment state grid with boundary padding
        q_function: Pre-loaded Q-table for collision avoidance
    """

    def __init__(self, agv_starts=None, tasks=None, output_file='yaml/ga_output.yaml'):
        """
        Initialize the path planner.

        Args:
            agv_starts: List of AGV starting positions [[x,y], [x,y], ...]
            tasks: List of task dictionaries [{'pickup':[x,y], 'dropoff':[x,y]}, ...]
            output_file: Output file path for saving planning results
        """
        self.output_file = output_file

        # Load map configuration
        with open(_MAP_FILE, 'r', encoding='utf-8') as f:
            map_data = yaml.load(f, Loader=yaml.Loader)

        self.col = map_data['map']['width']
        self.row = map_data['map']['height']
        self.agent_size = len(agv_starts)

        # Extract obstacle positions from map
        self.obs = []
        for y in range(self.row):
            for x in range(self.col):
                if map_data['map']['data'][y][x] == 1:
                    self.obs.append([x, y])

        # Initialize task tracking arrays
        self.task_pos = np.zeros(self.agent_size, dtype=int)
        self.pick_or_place = np.full(self.agent_size, 0, dtype=int)

        # Initialize path storage
        self.task_list = {}
        self.agent_pos = {}

        for agent in range(self.agent_size):
            initial_pos = agv_starts[agent]
            self.task_list[str(agent)] = [initial_pos]
            self.agent_pos[str(agent)] = [initial_pos]

        # Store task definitions
        self.custom_tasks = tasks

        # Pre-assign tasks to AGVs based on ordered_tasks indexing
        # Task i is assigned to AGV i (direct mapping)
        for agent_idx in range(min(len(tasks), self.agent_size)):
            task = tasks[agent_idx]
            self.task_list[str(agent_idx)].append(task['pickup'])
            self.task_list[str(agent_idx)].append(task['dropoff'])

        # Initialize state grid with proper boundary padding
        self.state = np.full((self.col, self.row), -2, dtype=int)
        for obstacle in self.obs:
            self.state[obstacle[0], obstacle[1]] = -1

        # Add 2-layer boundary around the map for safety
        for _ in range(2):
            self.state = np.insert(self.state, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = np.insert(self.state.T, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = self.state.T

        # Load pre-trained Q-function for collision avoidance
        self.q_function = np.load('yaml/qfunction.npy')

    def astar(self, t, agent, goal, start):
        """
        Perform A* path planning from start to goal position.

        Args:
            t: Current time step
            agent: Agent ID
            goal: Target position [x, y]
            start: Starting position [x, y]

        Returns:
            bool: True if path found successfully, False otherwise
        """
        a_star = AStar(self.col, self.row, goal, start, False, self.obs)
        path = a_star.main()

        if path:
            for node in path:
                self.agent_pos[str(agent)].append([node.x, node.y])

            self.agent_pos[str(agent)].append([path[-1].x, path[-1].y])

            return True
        return False

    def get_state(self, t, agent):
        """
        Extract local observation state for Q-learning collision avoidance.

        The state consists of:
        - Local 5x5 grid view centered on agent
        - Relative positions of other AGVs
        - Obstacle positions
        - Direction to goal
        - Potential passing conflicts

        Args:
            t: Current time step
            agent: Agent ID

        Returns:
            tuple: (local_state, rotation_angle, passing_info)
                - local_state: 5x5 numpy array representing local view
                - rotation_angle: Rotation needed to align with movement direction
                - passing_info: Array indicating potential conflicts with other AGVs
        """
        passing = np.zeros(len(ACTION_SET) - 1, dtype=int)
        state_ = np.copy(self.state)

        # Calculate current movement direction
        a = np.array(self.agent_pos[str(agent)][t + 1]) - \
            np.array(self.agent_pos[str(agent)][t])
        idx = 0
        for i in range(len(ACTION_SET)):
            if np.all(ACTION_SET[i] - a == 0):
                idx = i
                break

        # Detect nearby AGVs and potential conflicts
        for i in range(self.agent_size):
            if i != agent:
                for j in range(len(ACTION_SET) - 1):
                    temp = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[j]
                    if (np.all(np.array(self.agent_pos[str(i)][t]) - temp == 0) and
                            np.all(np.array(self.agent_pos[str(i)][t + 1]) -
                                   np.array(self.agent_pos[str(agent)][t]) == 0)):
                        passing[j] = 1
                state_[self.agent_pos[str(i)][t + 1][0] + 2,
                       self.agent_pos[str(i)][t + 1][1] + 2] = 0

        # Extract 5x5 local view
        local_state = state_[
            self.agent_pos[str(agent)][t][0]:self.agent_pos[str(agent)][t][0] + 5,
            self.agent_pos[str(agent)][t][1]:self.agent_pos[str(agent)][t][1] + 5
        ]

        # Rotate local state to align with movement direction
        for _ in range(idx):
            local_state = np.rot90(local_state, -1)

        return local_state, idx, passing

    def main(self):
        """
        Execute main planning loop.

        The planning process alternates between:
        1. Task assignment and A* path planning
        2. Q-learning-based collision avoidance
        3. Termination condition checking
        4. Solution validation

        Returns:
            tuple: (makespan, elapsed_time, distance)
                - makespan: Total completion time (max over all agents)
                - elapsed_time: Computation time in seconds
                - distance: Placeholder for compatibility (always False)
        """
        t = 0
        start_time = time.time()

        while True:
            # ========================================================================
            # Phase 1: Task Assignment and A* Path Planning
            # ========================================================================
            for agent in range(self.agent_size):
                # Ensure path buffer has sufficient length
                while len(self.agent_pos[str(agent)]) <= t + BASE_STOP_TIME:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])
                target_pos = np.array(self.task_list[str(agent)][self.task_pos[agent]])

                # Check if agent has reached current target
                if np.all(current_pos - target_pos == 0):
                    # Advance to next target if available
                    if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                        next_target = self.task_list[str(agent)][self.task_pos[agent] + 1]
                        start_pos = self.agent_pos[str(agent)][t + BASE_STOP_TIME]

                        # Plan path to next target
                        success = self.astar(t, agent, next_target, start_pos)

                        # Only advance task position if path planning succeeded
                        if success:
                            self.task_pos[agent] += 1
                    else:
                        # All tasks completed, remain at final position
                        self.agent_pos[str(agent)].append(
                            self.agent_pos[str(agent)][t + BASE_STOP_TIME]
                        )

            # ========================================================================
            # Phase 2: Q-learning-based Collision Avoidance
            # ========================================================================
            for agent in range(self.agent_size):
                # Ensure path buffer has sufficient length
                while len(self.agent_pos[str(agent)]) <= t + 1:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                # Get local observation state
                local_state, rotate, passing = self.get_state(t, agent)

                # Calculate goal direction
                goal = np.array(self.task_list[str(agent)][self.task_pos[agent]]) - \
                       np.array(self.agent_pos[str(agent)][t])

                # Determine preferred direction
                if abs(goal[0]) >= abs(goal[1]):
                    muki = 0 if goal[0] < 0 else 2
                else:
                    muki = 1 if goal[1] < 0 else 3
                muki = (muki - rotate) % 4

                # Encode state for Q-table lookup
                agv_node = '0000000000000'
                obs_node = '0000'
                passing_ = np.zeros(len(ACTION_SET) - 1, dtype=int)

                # Encode AGV positions in local view
                for i in range(len(OBSERVATION_NODES)):
                    if local_state[OBSERVATION_NODES[i][0]][OBSERVATION_NODES[i][1]] == 0:
                        agv_node = agv_node[:i] + '1' + agv_node[i + 1:]

                # Encode obstacle positions
                for i in range(len(OBSTACLE_NODES)):
                    if local_state[OBSTACLE_NODES[i][0]][OBSTACLE_NODES[i][1]] == -1:
                        obs_node = obs_node[:i] + '1' + obs_node[i + 1:]

                # Encode goal direction and passing information
                goal_node = format(muki, '02b')
                for p in range(len(passing)):
                    if passing[p] == 1:
                        passing_[(4 + p - rotate) % len(passing)] = 1

                passing__ = passing_[0] * 8 + passing_[1] * 4 + \
                            passing_[2] * 2 + passing_[3]
                passing__ = format(passing__, '04b')
                state_number = int(agv_node + obs_node + goal_node + passing__, 2)

                # Query Q-table and select best action
                q = self.q_function[state_number]
                max_value = np.max(q)
                max_index = [a for a in range(len(q)) if q[a] == max_value]
                action = max_index[np.random.randint(0, len(max_index))]

                # Apply collision avoidance maneuver if necessary
                if (action != 0 and
                        not np.all(np.array(self.agent_pos[str(agent)][t]) -
                                   np.array(self.agent_pos[str(agent)][t + 1]) == 0)):
                    if action != 4:
                        action = (action + rotate) % 4

                    # Truncate path and apply new action
                    self.agent_pos[str(agent)] = self.agent_pos[str(agent)][:t + 1]
                    new_pos = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[action]
                    self.agent_pos[str(agent)].append(new_pos.tolist())

                    # Replan from new position
                    self.astar(t, agent,
                               self.task_list[str(agent)][self.task_pos[agent]],
                               new_pos.tolist())

            t += 1

            # ========================================================================
            # Phase 3: Termination Condition Check
            # ========================================================================
            all_completed = True
            for agent in range(self.agent_size):
                # Check if agent has unfinished tasks
                if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                    all_completed = False
                    break

                # Verify agent has reached final destination
                current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])
                final_goal = np.array(self.task_list[str(agent)][-1])
                if not np.all(current_pos == final_goal):
                    all_completed = False
                    break

            if all_completed:
                elapsed_time = time.time() - start_time

                # ====================================================================
                # Phase 4: Solution Validation (Post-verification)
                # ====================================================================
                all_tasks_valid = True

                for task_idx, task in enumerate(self.custom_tasks):
                    pickup = tuple(task['pickup'])
                    dropoff = tuple(task['dropoff'])

                    # Identify which AGV was assigned this task
                    task_assigned_to_agent = -1

                    for agent in range(self.agent_size):
                        agent_tasks = self.task_list[str(agent)]
                        for i in range(len(agent_tasks) - 1):
                            if (tuple(agent_tasks[i]) == pickup and
                                    tuple(agent_tasks[i + 1]) == dropoff):
                                task_assigned_to_agent = agent
                                break

                    if task_assigned_to_agent == -1:
                        all_tasks_valid = False
                        print(f"[WARN] Task{task_idx} not assigned to any AGV: "
                              f"pickup={pickup}, dropoff={dropoff}")
                        break

                    # Verify task execution by assigned AGV
                    path = self.agent_pos[str(task_assigned_to_agent)]
                    pickup_time = -1
                    dropoff_time = -1

                    for step_idx, pos in enumerate(path):
                        if tuple(pos) == pickup and pickup_time == -1:
                            pickup_time = step_idx
                        if (tuple(pos) == dropoff and
                                pickup_time != -1 and dropoff_time == -1):
                            dropoff_time = step_idx
                            break

                    if pickup_time == -1:
                        all_tasks_valid = False
                        print(f"[WARN] Task{task_idx} (AGV{task_assigned_to_agent}) "
                              f"did not reach pickup: {pickup}")
                        break

                    if dropoff_time == -1:
                        all_tasks_valid = False
                        print(f"[WARN] Task{task_idx} (AGV{task_assigned_to_agent}) "
                              f"did not reach dropoff: {dropoff}")
                        break

                    # Validate停留 time constraints at pickup point
                    pickup_stay_count = 0
                    for step_idx in range(pickup_time, len(path)):
                        if tuple(path[step_idx]) == pickup:
                            pickup_stay_count += 1
                        else:
                            break

                    required_stay = BASE_STOP_TIME + LOAD_UNLOAD_TIME
                    if pickup_stay_count < required_stay:
                        all_tasks_valid = False
                        print(f"[WARN] Task{task_idx} (AGV{task_assigned_to_agent}) "
                              f"insufficient pickup停留 time: {pickup_stay_count} steps "
                              f"(required: {required_stay} steps)")
                        break

                    # Validate停留 time constraints at dropoff point
                    dropoff_stay_count = 0
                    for step_idx in range(dropoff_time, len(path)):
                        if tuple(path[step_idx]) == dropoff:
                            dropoff_stay_count += 1
                        else:
                            break

                    if dropoff_stay_count < required_stay:
                        all_tasks_valid = False
                        print(f"[WARN] Task{task_idx} (AGV{task_assigned_to_agent}) "
                              f"insufficient dropoff停留 time: {dropoff_stay_count} steps "
                              f"(required: {required_stay} steps)")
                        break

                # Reject infeasible solutions
                if not all_tasks_valid:
                    print("[ERROR] Incomplete tasks detected, returning 0 steps "
                          "(infeasible solution)")
                    return 0, elapsed_time, False

                # Save valid solution to file
                output_data = {'schedule': {}}
                for agent in range(self.agent_size):
                    schedule = []
                    for step, pos in enumerate(self.agent_pos[str(agent)]):
                        schedule.append({'t': step, 'x': int(pos[0]), 'y': int(pos[1])})
                    output_data['schedule'][f'agent{agent}'] = schedule

                with open(self.output_file, 'w') as f:
                    yaml.dump(output_data, f)

                # Calculate makespan (maximum completion time across all agents)
                actual_steps = []
                for agent in range(self.agent_size):
                    path_length = len(self.agent_pos[str(agent)]) - 1
                    actual_steps.append(path_length)

                makespan = max(actual_steps)
                return makespan, elapsed_time, False

            # Timeout check
            if t > MAX_PLANNING_STEPS:
                elapsed_time = time.time() - start_time

                # Save partial solution for debugging
                output_data = {'schedule': {}}
                for agent in range(self.agent_size):
                    schedule = []
                    for step, pos in enumerate(self.agent_pos[str(agent)]):
                        schedule.append({'t': step, 'x': int(pos[0]), 'y': int(pos[1])})
                    output_data['schedule'][f'agent{agent}'] = schedule

                with open(self.output_file, 'w') as f:
                    yaml.dump(output_data, f)

                return 0, elapsed_time, False


if __name__ == '__main__':
    # Test execution with instance configuration
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    planning = Planning(
        agv_starts=instance['agv_starts'],
        tasks=instance['tasks'],
        output_file='yaml/output_test.yaml'
    )

    step, elapsed_time, dist = planning.main()
    print(f"Completion steps: {step}")
    print(f"Computation time: {elapsed_time:.4f} seconds")
