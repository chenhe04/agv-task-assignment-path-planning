# verify_heuristic_solution.py
"""
Verification Tool: Check if Heuristic Solution Satisfies MIP Constraints

Purpose:
    Load heuristic solution (from YAML) and verify it against all
    MIP model constraints to debug discrepancies between MIP and heuristic methods.

Usage:
    python verify_heuristic_solution.py [--solution-file PATH]

Configuration:
    - SOLUTION_FILE: Path to heuristic solution file (default: yaml/output_enum.yaml)
    - INSTANCE_YAML: Path to problem instance file
    - MAP_FILE: Path to map file
    - LOAD_UNLOAD_TIME: Loading/unloading duration (time steps)
"""

import yaml
import numpy as np
import os
import sys
import argparse

# ========== Configuration ==========
SOLUTION_FILE = 'yaml/output_ga_large.yaml'  # Default solution file
INSTANCE_YAML = 'instance.yaml'  # Problem instance
LOAD_UNLOAD_TIME = 1  # Loading/unloading time (steps)

# Map file priority
CUSTOM_MAP_PRIORITY = [
    'custom_map.yaml',
    'map_small_traditional_aisle.yaml',
    'map_medium_traditional_aisle.yaml',
    'map_large_traditional_aisle.yaml'
]


# ==================================


def load_map_file():
    """Auto-detect map file."""
    for map_file in CUSTOM_MAP_PRIORITY:
        if os.path.exists(map_file):
            return map_file
    raise FileNotFoundError(
        "Map file not found! Expected: custom_map.yaml or map_{small/medium/large}_traditional_aisle.yaml"
    )


def load_instance():
    """Load problem instance (AGV starts, tasks)."""
    with open(INSTANCE_YAML, 'r', encoding='utf-8') as f:
        instance = yaml.load(f, Loader=yaml.Loader)
    return instance


def load_map_data(map_file):
    """Load map dimensions and obstacles."""
    with open(map_file, 'r', encoding='utf-8') as f:
        map_data = yaml.load(f, Loader=yaml.Loader)

    width = map_data['map']['width']
    height = map_data['map']['height']

    obstacles = []
    for y in range(height):
        for x in range(width):
            if map_data['map']['data'][y][x] == 1:
                obstacles.append([x, y])

    return width, height, obstacles


def load_heuristic_solution(yaml_file):
    """Load heuristic solution from YAML file."""
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Heuristic solution not found: {yaml_file}")

    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    schedule = data['schedule']
    paths = {}
    for agent_key, agent_schedule in schedule.items():
        agv_id = int(agent_key.replace('agent', ''))
        path = [(step['x'], step['y']) for step in agent_schedule]
        paths[agv_id] = path

    return paths


def compute_shortest_path_distance(start, goal, obstacles, width, height):
    """Compute Manhattan distance (simplified, can use A* if needed)."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def find_task_execution(path, pickup, dropoff, required_stay):
    """
    Find valid task execution in a path.

    A valid task execution requires:
    1. AGV stays at pickup for >= required_stay consecutive steps (actual loading)
    2. After pickup, AGV stays at dropoff for >= required_stay consecutive steps (actual unloading)
    3. AGV remains at dropoff until the end of the path

    Pass-through visits (staying < required_stay) are NOT considered as task execution.

    Args:
        path: List of (x, y) positions
        pickup: Pickup position tuple
        dropoff: Dropoff position tuple
        required_stay: Minimum consecutive stay steps for loading/unloading

    Returns:
        tuple: (pickup_time, dropoff_time) or (-1, -1) if not valid
    """
    pickup_time = -1
    for t in range(len(path)):
        if path[t] == pickup:
            stay = 0
            for k in range(t, len(path)):
                if path[k] == pickup:
                    stay += 1
                else:
                    break
            if stay >= required_stay:
                pickup_time = t
                break

    if pickup_time == -1:
        return -1, -1

    dropoff_time = -1
    search_start = pickup_time + required_stay
    for t in range(search_start, len(path)):
        if path[t] == dropoff:
            stay = 0
            for k in range(t, len(path)):
                if path[k] == dropoff:
                    stay += 1
                else:
                    break
            if stay >= required_stay:
                stays_until_end = all(path[k] == dropoff for k in range(t, len(path)))
                if stays_until_end:
                    dropoff_time = t
                    break

    if dropoff_time == -1:
        return -1, -1

    return pickup_time, dropoff_time


def verify_initial_position(paths, agv_starts):
    """Constraint 1: Initial position at t=0."""
    print("\n" + "=" * 70)
    print("【Constraint 1】Initial Position (t=0)")
    print("=" * 70)

    violations = []
    n_agvs = len(paths)

    for i in range(n_agvs):
        expected_pos = tuple(agv_starts[i])
        actual_pos = paths[i][0] if len(paths[i]) > 0 else None

        if actual_pos != expected_pos:
            violation = f"AGV{i}: starts at {actual_pos}, expected {expected_pos}"
            violations.append(violation)
            print(f"  ✗ AGV{i}: {actual_pos} (expected {expected_pos})")
        else:
            print(f"  ✓ AGV{i}: {actual_pos}")

    if len(violations) == 0:
        print(f"  ✓ All AGVs start at correct positions")

    return violations


def verify_movement_continuity(paths):
    """Constraint 2: Movement continuity (adjacent moves only)."""
    print("\n" + "=" * 70)
    print("【Constraint 2】Movement Continuity")
    print("=" * 70)

    violations = []
    n_agvs = len(paths)

    for i in range(n_agvs):
        for t in range(len(paths[i]) - 1):
            dx = abs(paths[i][t + 1][0] - paths[i][t][0])
            dy = abs(paths[i][t + 1][1] - paths[i][t][1])

            if dx + dy > 1:
                violation = f"AGV{i} at t={t}: jump from {paths[i][t]} to {paths[i][t + 1]} (dx+dy={dx + dy})"
                violations.append(violation)
                if len(violations) <= 10:
                    print(f"  ✗ {violation}")

    if len(violations) == 0:
        print(f"  ✓ All movements valid (adjacent or stay)")
    else:
        print(f"  ⚠ {len(violations)} movement violations")
        if len(violations) > 10:
            print(f"    (showing first 10, total: {len(violations)})")

    return violations


def verify_vertex_conflicts(paths):
    """Constraint 3: No vertex conflicts."""
    print("\n" + "=" * 70)
    print("【Constraint 3】Vertex Conflicts")
    print("=" * 70)

    violations = []
    n_agvs = len(paths)
    max_time = max(len(p) for p in paths.values())

    for t in range(max_time):
        positions = {}
        for i in range(n_agvs):
            if t < len(paths[i]):
                pos = paths[i][t]
            else:
                pos = paths[i][-1]

            if pos in positions:
                violation = f"t={t}: AGV{i} and AGV{positions[pos]} both at {pos}"
                violations.append(violation)
                if len(violations) <= 10:
                    print(f"  ✗ {violation}")
            else:
                positions[pos] = i

    if len(violations) == 0:
        print(f"  ✓ No vertex conflicts")
    else:
        print(f"  ⚠ {len(violations)} vertex conflicts")
        if len(violations) > 10:
            print(f"    (showing first 10, total: {len(violations)})")

    return violations


def verify_edge_conflicts(paths):
    """Constraint 4: No edge conflicts (head-to-head collisions)."""
    print("\n" + "=" * 70)
    print("【Constraint 4】Edge Conflicts")
    print("=" * 70)

    violations = []
    n_agvs = len(paths)
    max_time = max(len(p) for p in paths.values())

    for t in range(1, max_time):
        for i in range(n_agvs):
            for j in range(i + 1, n_agvs):
                pos_i_prev = paths[i][t - 1] if t - 1 < len(paths[i]) else paths[i][-1]
                pos_i_curr = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                pos_j_prev = paths[j][t - 1] if t - 1 < len(paths[j]) else paths[j][-1]
                pos_j_curr = paths[j][t] if t < len(paths[j]) else paths[j][-1]

                if pos_i_prev == pos_j_curr and pos_i_curr == pos_j_prev:
                    violation = f"t={t}: AGV{i} and AGV{j} head-to-head {pos_i_prev}↔{pos_i_curr}"
                    violations.append(violation)
                    if len(violations) <= 10:
                        print(f"  ✗ {violation}")

    if len(violations) == 0:
        print(f"  ✓ No edge conflicts")
    else:
        print(f"  ⚠ {len(violations)} edge conflicts")
        if len(violations) > 10:
            print(f"    (showing first 10, total: {len(violations)})")

    return violations


def infer_task_assignment(paths, tasks):
    """Infer task assignment from paths using valid task execution detection."""
    task_assignment = {}
    required_stay = LOAD_UNLOAD_TIME + 1

    for task_idx, task in enumerate(tasks):
        pickup = tuple(task['pickup'])
        dropoff = tuple(task['dropoff'])

        assigned_agvs = []
        for agv_id in paths.keys():
            pickup_time, dropoff_time = find_task_execution(
                paths[agv_id], pickup, dropoff, required_stay
            )
            if pickup_time != -1:
                assigned_agvs.append(agv_id)

        if len(assigned_agvs) == 1:
            task_assignment[task_idx] = assigned_agvs[0]

    return task_assignment


def verify_task_assignment(paths, tasks):
    """Constraint 5: Each task assigned to exactly one AGV."""
    print("\n" + "=" * 70)
    print("【Constraint 5】Task Assignment")
    print("=" * 70)

    violations = []
    n_tasks = len(tasks)
    completed_count = 0
    required_stay = LOAD_UNLOAD_TIME + 1

    for task_idx in range(n_tasks):
        task = tasks[task_idx]
        pickup = tuple(task['pickup'])
        dropoff = tuple(task['dropoff'])

        assigned_agvs = []
        for agv_id in paths.keys():
            pickup_time, dropoff_time = find_task_execution(
                paths[agv_id], pickup, dropoff, required_stay
            )
            if pickup_time != -1:
                assigned_agvs.append(agv_id)

        if len(assigned_agvs) == 0:
            violation = f"Task{task_idx}: NOT ASSIGNED (pickup:{pickup}, dropoff:{dropoff})"
            violations.append(violation)
            print(f"  ✗ Task{task_idx}: NOT ASSIGNED")

            # Show partial visits for debugging
            for agv_id in paths.keys():
                if pickup in paths[agv_id]:
                    print(f"    - AGV{agv_id} visited pickup {pickup} at t={paths[agv_id].index(pickup)}")
                if dropoff in paths[agv_id]:
                    print(f"    - AGV{agv_id} visited dropoff {dropoff} at t={paths[agv_id].index(dropoff)}")
        elif len(assigned_agvs) > 1:
            violation = f"Task{task_idx}: assigned to multiple AGVs {assigned_agvs}"
            violations.append(violation)
            print(f"  ✗ Task{task_idx}: multiple AGVs {assigned_agvs}")
        else:
            completed_count += 1
            print(f"  ✓ Task{task_idx} → AGV{assigned_agvs[0]}")

    # Check each AGV has at most one task
    task_assignment = infer_task_assignment(paths, tasks)
    agv_task_count = {}
    for task_idx, agv_id in task_assignment.items():
        if agv_id not in agv_task_count:
            agv_task_count[agv_id] = []
        agv_task_count[agv_id].append(task_idx)

    for agv_id, task_list in agv_task_count.items():
        if len(task_list) > 1:
            violation = f"AGV{agv_id}: assigned to multiple tasks {task_list}"
            violations.append(violation)
            print(f"  ✗ AGV{agv_id}: multiple tasks {task_list}")

    print(f"\n  Summary: {completed_count}/{n_tasks} tasks completed")

    if len(violations) == 0:
        print(f"  ✓ Valid task assignment (all tasks completed)")
    else:
        print(f"  ⚠ {len(violations)} tasks not completed")

    return violations, task_assignment


def verify_visit_constraints(paths, tasks, task_assignment):
    """Constraint 6: Visit pickup and dropoff points."""
    print("\n" + "=" * 70)
    print("【Constraint 6】Visit Pickup and Dropoff")
    print("=" * 70)

    violations = []
    required_stay = LOAD_UNLOAD_TIME + 1

    for task_idx, task in enumerate(tasks):
        if task_idx not in task_assignment:
            continue

        agv_id = task_assignment[task_idx]
        pickup = tuple(task['pickup'])
        dropoff = tuple(task['dropoff'])

        pickup_time, dropoff_time = find_task_execution(
            paths[agv_id], pickup, dropoff, required_stay
        )

        if pickup_time == -1:
            violation = f"Task{task_idx} (AGV{agv_id}): pickup {pickup} not visited with sufficient stay"
            violations.append(violation)
            print(f"  ✗ {violation}")
        else:
            print(f"  ✓ Task{task_idx} (AGV{agv_id}): visits pickup at t={pickup_time}")

        if dropoff_time == -1:
            violation = f"Task{task_idx} (AGV{agv_id}): dropoff {dropoff} not visited with sufficient stay"
            violations.append(violation)
            print(f"  ✗ {violation}")
        else:
            print(f"  ✓ Task{task_idx} (AGV{agv_id}): visits dropoff at t={dropoff_time}")

        if pickup_time != -1 and dropoff_time != -1:
            print(f"  ✓ Task{task_idx} (AGV{agv_id}): visits both points")

    if len(violations) == 0:
        print(f"  ✓ All tasks visit required points")

    return violations


def verify_stay_duration(paths, tasks, task_assignment):
    """Constraint 7: Loading/unloading stay duration."""
    print("\n" + "=" * 70)
    print("【Constraint 7】Stay Duration (Loading/Unloading)")
    print("=" * 70)
    print(f"  Required continuous stay: {LOAD_UNLOAD_TIME + 1} steps")

    violations = []
    required_stay = LOAD_UNLOAD_TIME + 1

    for task_idx, task in enumerate(tasks):
        if task_idx not in task_assignment:
            continue

        agv_id = task_assignment[task_idx]
        pickup = tuple(task['pickup'])
        dropoff = tuple(task['dropoff'])

        pickup_time, dropoff_time = find_task_execution(
            paths[agv_id], pickup, dropoff, required_stay
        )

        if pickup_time != -1:
            pickup_stay = 0
            for k in range(pickup_time, len(paths[agv_id])):
                if paths[agv_id][k] == pickup:
                    pickup_stay += 1
                else:
                    break
            print(f"  ✓ Task{task_idx} (AGV{agv_id}): pickup stay {pickup_stay} steps (from t={pickup_time})")

        if dropoff_time != -1:
            dropoff_stay = len(paths[agv_id]) - dropoff_time
            print(f"  ✓ Task{task_idx} (AGV{agv_id}): dropoff stay {dropoff_stay} steps (from t={dropoff_time})")
        else:
            violation = f"Task{task_idx} (AGV{agv_id}): no valid dropoff execution"
            violations.append(violation)
            print(f"  ✗ {violation}")

    if len(violations) == 0:
        print(f"  ✓ All stay requirements satisfied")

    return violations


def verify_temporal_constraints(paths, tasks, task_assignment):
    """Constraint 8: Temporal order (pickup before dropoff)."""
    print("\n" + "=" * 70)
    print("【Constraint 8】Temporal Order")
    print("=" * 70)

    violations = []
    required_stay = LOAD_UNLOAD_TIME + 1

    for task_idx, task in enumerate(tasks):
        if task_idx not in task_assignment:
            continue

        agv_id = task_assignment[task_idx]
        pickup = tuple(task['pickup'])
        dropoff = tuple(task['dropoff'])

        pickup_time, dropoff_time = find_task_execution(
            paths[agv_id], pickup, dropoff, required_stay
        )

        if pickup_time != -1 and dropoff_time != -1:
            if pickup_time < dropoff_time:
                print(f"  ✓ Task{task_idx}: pickup@t={pickup_time} < dropoff@t={dropoff_time}")
            else:
                violation = f"Task{task_idx}: pickup@t={pickup_time} >= dropoff@t={dropoff_time}"
                violations.append(violation)
                print(f"  ✗ {violation}")

    if len(violations) == 0:
        print(f"  ✓ All temporal orders correct")

    return violations


def main():
    """Main verification process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Verify heuristic solution against MIP constraints')
    parser.add_argument('--solution-file', type=str, default=None, help='Path to heuristic solution YAML file')
    args = parser.parse_args()

    # Override configuration if arguments provided
    if args.solution_file:
        global SOLUTION_FILE
        SOLUTION_FILE = args.solution_file

    print("=" * 70)
    print("HEURISTIC SOLUTION VERIFICATION TOOL")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    try:
        instance = load_instance()
        map_file = load_map_file()
        width, height, obstacles = load_map_data(map_file)
        paths = load_heuristic_solution(SOLUTION_FILE)

        print(f"  ✓ Instance: {len(instance['agv_starts'])} AGVs, {len(instance['tasks'])} tasks")
        print(f"  ✓ Map: {width}x{height}, {len(obstacles)} obstacles")
        print(f"  ✓ Solution loaded from: {SOLUTION_FILE}")
        print(f"  ✓ Makespan: {max(len(p) for p in paths.values()) - 1} steps")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run verifications
    all_violations = []

    v1 = verify_initial_position(paths, instance['agv_starts'])
    all_violations.extend([('initial_position', v) for v in v1])

    v2 = verify_movement_continuity(paths)
    all_violations.extend([('movement', v) for v in v2])

    v3 = verify_vertex_conflicts(paths)
    all_violations.extend([('vertex_conflict', v) for v in v3])

    v4 = verify_edge_conflicts(paths)
    all_violations.extend([('edge_conflict', v) for v in v4])

    v5, task_assignment = verify_task_assignment(paths, instance['tasks'])
    all_violations.extend([('task_assignment', v) for v in v5])

    v6 = verify_visit_constraints(paths, instance['tasks'], task_assignment)
    all_violations.extend([('visit', v) for v in v6])

    v7 = verify_stay_duration(paths, instance['tasks'], task_assignment)
    all_violations.extend([('stay_duration', v) for v in v7])

    v8 = verify_temporal_constraints(paths, instance['tasks'], task_assignment)
    all_violations.extend([('temporal', v) for v in v8])

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    violation_by_type = {}
    for vtype, vdesc in all_violations:
        if vtype not in violation_by_type:
            violation_by_type[vtype] = 0
        violation_by_type[vtype] += 1

    print(f"\nTotal violations: {len(all_violations)}")

    if len(violation_by_type) > 0:
        print(f"\nViolations by type:")
        for vtype, count in violation_by_type.items():
            print(f"  • {vtype}: {count}")

    if len(all_violations) == 0:
        print(f"\n{'=' * 70}")
        print("✅ SUCCESS: HEURISTIC SOLUTION SATISFIES ALL MIP CONSTRAINTS!")
        print(f"{'=' * 70}")
        print(f"\nImplications:")
        print(f"  • The heuristic solution is feasible for the MIP model")
        print(f"  • Makespan: {max(len(p) for p in paths.values()) - 1} steps")
        print(f"  • All {len(instance['tasks'])} tasks are completed")
        print(f"\nIf MIP gives worse result, possible causes:")
        print(f"  1. MIP solver didn't converge → Increase TIME_LIMIT")
        print(f"  2. MIP_GAP too loose → Set MIP_GAP = 0.0")
        print(f"  3. Stay constraints over-restrictive → Check formulation")
        print(f"  4. Time horizon T too small → Increase T estimation")
    else:
        print(f"\n{'=' * 70}")
        print(f"❌ FAILURE: HEURISTIC SOLUTION VIOLATES {len(all_violations)} CONSTRAINTS")
        print(f"{'=' * 70}")
        print(f"\n⚠️  CRITICAL FINDING:")

        # Count uncompleted tasks
        uncompleted_tasks = [v for v in all_violations if v[0] == 'task_assignment']
        if uncompleted_tasks:
            print(f"  • Heuristic method did NOT complete all tasks!")
            print(f"  • Uncompleted: {len(uncompleted_tasks)} out of {len(instance['tasks'])} tasks")
            print(
                f"  • Completion rate: {(len(instance['tasks']) - len(uncompleted_tasks)) / len(instance['tasks']) * 100:.1f}%")
            print(f"  • The makespan ({max(len(p) for p in paths.values()) - 1} steps) is for PARTIAL solution")
            print(f"\nThis means:")
            print(f"  • You CANNOT directly compare heuristic vs MIP results")
            print(f"  • MIP may be completing ALL tasks while heuristic only completes some")
            print(f"  • To fairly compare, ensure both methods complete the SAME tasks")

        print(f"\nThe two methods have different constraint sets or interpretations.")
        print(f"\nFirst 10 violations:")
        for i, (vtype, vdesc) in enumerate(all_violations[:10]):
            print(f"  {i + 1}. [{vtype}] {vdesc}")
        if len(all_violations) > 10:
            print(f"  ... ({len(all_violations) - 10} more)")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
