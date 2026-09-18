# step3_run_mip.py
"""
Optimal Task Assignment and Path Planning using Gurobi MIP

Modes:
    Mode 1: Relaxed MIP (without conflict constraints) - provides lower bound
    Mode 2: Complete MIP (with vertex and edge conflict avoidance)

Output:
    - Optimal task assignment
    - Collision-free paths (Mode 2) or relaxed paths (Mode 1)
    - Makespan minimization
"""

import yaml
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from astar_class import AStar
import time
import os

# ========== Configuration ==========
MIP_MODE = 1
# 1=Relaxed MIP, 2=Complete MIP
TIME_LIMIT = 6000  # Gurobi time limit (seconds)
LOAD_UNLOAD_TIME = 1  # Loading/unloading duration (time steps)
# ==================================

# ========== Map File Detection ==========
CUSTOM_MAP_PRIORITY = [
    'custom_map.yaml',
    'map_small_traditional_aisle.yaml',
    'map_medium_traditional_aisle.yaml',
    'map_large_traditional_aisle.yaml'
]

MAP_FILE = None
for map_file in CUSTOM_MAP_PRIORITY:
    if os.path.exists(map_file):
        MAP_FILE = map_file
        break

if MAP_FILE is None:
    raise FileNotFoundError(
        "Map file not found! Please run step1_create_instance.py first.\n"
        "Expected files: custom_map.yaml or map_{small/medium/large}_traditional_aisle.yaml"
    )
else:
    print(f"[Auto-detect] Using map file: {MAP_FILE}")


# ==================================


def load_map_and_obstacles():
    """Load map dimensions and obstacle positions from YAML file."""
    with open(MAP_FILE, 'r', encoding='utf-8') as f:
        map_data = yaml.load(f, Loader=yaml.Loader)

    width = map_data['map']['width']
    height = map_data['map']['height']

    obstacles = []
    for y in range(height):
        for x in range(width):
            if map_data['map']['data'][y][x] == 1:
                obstacles.append([x, y])

    return width, height, obstacles


def get_astar_path(start, goal, obstacles, width, height):
    """
    Generate continuous path using A* algorithm.

    Args:
        start: [x, y] starting position
        goal: [x, y] goal position
        obstacles: List of obstacle coordinates
        width: Map width
        height: Map height

    Returns:
        List of (x, y) tuples representing the path, or empty list if no path found
    """
    if start == goal:
        return [tuple(start)]

    try:
        astar = AStar(
            cols=width, rows=height,
            start=[start[0], start[1]],
            end=[goal[0], goal[1]],
            obstacle_ratio=False,
            obstacle_list=obstacles
        )
        path_nodes = astar.main()
        if path_nodes:
            path = [(node.x, node.y) for node in path_nodes]

            if path and (path[0][0] != start[0] or path[0][1] != start[1]):
                path.reverse()

            if path and (path[0][0] != start[0] or path[0][1] != start[1]):
                path.insert(0, tuple(start))

            if path and (path[-1][0] != goal[0] or path[-1][1] != goal[1]):
                path.append(tuple(goal))

            valid = True
            for i in range(len(path) - 1):
                dx = abs(path[i + 1][0] - path[i][0])
                dy = abs(path[i + 1][1] - path[i][1])
                if dx + dy != 1:
                    valid = False
                    break

            if valid:
                return path
            else:
                return []
    except Exception as e:
        print(f"    A* error: {start} -> {goal}: {e}")

    return []


def compute_shortest_path_distance(start, goal, obstacles, width, height):
    """Compute shortest path distance using A* (returns number of steps)."""
    path = get_astar_path(start, goal, obstacles, width, height)
    return len(path) - 1 if path else float('inf')


def compute_distance_matrix(agv_starts, tasks, obstacles, width, height):
    """Compute distance matrix between all relevant points (AGV starts, pickups, dropoffs)."""
    all_points = []
    for start in agv_starts:
        all_points.append(tuple(start))
    for task in tasks:
        all_points.append(tuple(task['pickup']))
        all_points.append(tuple(task['dropoff']))

    all_points = list(dict.fromkeys(all_points))
    n = len(all_points)

    point_to_idx = {p: i for i, p in enumerate(all_points)}
    dist = np.full((n, n), np.inf)

    for i, start in enumerate(all_points):
        dist[i, i] = 0
        for j, goal in enumerate(all_points):
            if i != j and dist[i, j] == np.inf:
                d = compute_shortest_path_distance(
                    [start[0], start[1]],
                    [goal[0], goal[1]],
                    obstacles, width, height
                )
                if d != float('inf'):
                    dist[i, j] = d
                    dist[j, i] = d

    return dist, all_points, point_to_idx


def generate_paths_with_conflict_check(assignment, agv_starts, tasks, obstacles, width, height):
    """
    Generate paths based on assignment using A*, with conflict detection.
    Used in Mode 1 for post-hoc verification.
    """
    paths = {}
    agv_to_task = {}
    for task_id, agv_id in assignment.items():
        agv_to_task[agv_id] = task_id

    for agv_id in range(len(agv_starts)):
        if agv_id not in agv_to_task:
            paths[agv_id] = [tuple(agv_starts[agv_id])]
            continue

        task_id = agv_to_task[agv_id]
        task = tasks[task_id]
        start = agv_starts[agv_id]
        pickup = task['pickup']
        dropoff = task['dropoff']

        path1 = get_astar_path(start, pickup, obstacles, width, height)
        if not path1:
            paths[agv_id] = [tuple(start)]
            continue

        path2 = get_astar_path(pickup, dropoff, obstacles, width, height)
        if not path2:
            paths[agv_id] = path1
            continue

        full_path = []
        full_path.extend(path1)
        if path2 and full_path[-1] == path2[0]:
            full_path.extend(path2[1:])
        else:
            full_path.extend(path2)

        paths[agv_id] = full_path

    return paths


def detect_conflicts(paths):
    """
    Detect conflicts between AGV paths.

    Returns:
        List of conflicts with type ('vertex' or 'edge'), time, and involved AGVs
    """
    conflicts = []
    agv_ids = list(paths.keys())
    max_time = max(len(p) for p in paths.values())

    for t in range(max_time):
        positions = {}
        for agv_id in agv_ids:
            if t < len(paths[agv_id]):
                pos = paths[agv_id][t]
            else:
                pos = paths[agv_id][-1]

            if pos in positions:
                conflicts.append({
                    'type': 'vertex',
                    'time': t,
                    'agvs': [positions[pos], agv_id],
                    'position': pos
                })
            else:
                positions[pos] = agv_id

        if t > 0:
            for i_idx in range(len(agv_ids)):
                for j_idx in range(i_idx + 1, len(agv_ids)):
                    agv_i = agv_ids[i_idx]
                    agv_j = agv_ids[j_idx]

                    pos_i_prev = paths[agv_i][t - 1] if t - 1 < len(paths[agv_i]) else paths[agv_i][-1]
                    pos_i_curr = paths[agv_i][t] if t < len(paths[agv_i]) else paths[agv_i][-1]
                    pos_j_prev = paths[agv_j][t - 1] if t - 1 < len(paths[agv_j]) else paths[agv_j][-1]
                    pos_j_curr = paths[agv_j][t] if t < len(paths[agv_j]) else paths[agv_j][-1]

                    if pos_i_prev == pos_j_curr and pos_i_curr == pos_j_prev:
                        conflicts.append({
                            'type': 'edge',
                            'time': t,
                            'agvs': [agv_i, agv_j],
                            'positions': [pos_i_prev, pos_i_curr]
                        })

    return conflicts


def save_optimal_paths_yaml(paths, makespan, output_file='yaml/optimal_paths.yaml'):
    """Save optimal paths to YAML file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    max_len = max(len(p) for p in paths.values())

    schedule = {}
    for agv_id, path in paths.items():
        agent_schedule = []
        for t in range(max_len):
            if t < len(path):
                x, y = path[t]
            else:
                x, y = path[-1]
            agent_schedule.append({'t': t, 'x': x, 'y': y})
        schedule[f'agent{agv_id}'] = agent_schedule

    output_data = {
        'schedule': schedule,
        'metadata': {
            'makespan': makespan
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)

    print(f"  ✓ Saved: {output_file} (makespan: {makespan} steps)")


# ============================================================
# Mode 1: Relaxed MIP (without conflict constraints)
# ============================================================
def solve_mip_relaxed(agv_starts, tasks, obstacles, width, height,
                      save_yaml=True, output_file='yaml/optimal_paths_relaxed.yaml'):
    """
    Solve relaxed MIP without conflict constraints.

    This model provides a lower bound for the complete MIP:
    - Includes: movement continuity, task assignment, temporal constraints
    - Excludes: vertex conflict constraints, edge conflict constraints

    Theoretical guarantee: Makespan_relaxed ≤ Makespan_complete
    """
    n_agvs = len(agv_starts)
    n_tasks = len(tasks)

    print("\n" + "=" * 60)
    print("【Mode 1】Relaxed MIP (No Conflict Constraints)")
    print("=" * 60)
    print(f"  AGVs: {n_agvs}, Tasks: {n_tasks}")
    print(f"  Load/Unload Time: {LOAD_UNLOAD_TIME} step(s)")
    print(f"\n  Differences from Complete MIP:")
    print(f"    ✓ Included: Movement continuity, task assignment, temporal constraints")
    print(f"    ✗ Excluded: Vertex conflict, edge conflict constraints")
    print(f"\n  Theoretical Guarantee: Makespan_relaxed ≤ Makespan_full")

    # Estimate time horizon T
    dist_matrix, _, point_to_idx = compute_distance_matrix(
        agv_starts, tasks, obstacles, width, height
    )

    start_idx = [point_to_idx[tuple(p)] for p in agv_starts]
    pickup_idx = [point_to_idx[tuple(t['pickup'])] for t in tasks]
    dropoff_idx = [point_to_idx[tuple(t['dropoff'])] for t in tasks]

    max_dist = 0
    for k in range(n_agvs):
        for w in range(n_tasks):
            d = (dist_matrix[start_idx[k], pickup_idx[w]] +
                 LOAD_UNLOAD_TIME +
                 dist_matrix[pickup_idx[w], dropoff_idx[w]] +
                 LOAD_UNLOAD_TIME)
            if d != float('inf'):
                max_dist = max(max_dist, d)

    T = min(int(max_dist * 3) + 30, 150)
    print(f"\n  Estimated time horizon T = {T}")

    # Valid nodes (non-obstacle)
    valid_nodes = [(x, y) for x in range(width) for y in range(height)
                   if [x, y] not in obstacles]
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    n_nodes = len(valid_nodes)

    print(f"  Valid nodes: {n_nodes}")
    print(f"  Decision variables: ~{n_agvs * n_nodes * (T + 1):,}")

    print("\nStep 1: Building MIP model...")
    model = gp.Model("MAPF_Relaxed_MIP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', TIME_LIMIT)
    model.setParam('Threads', 4)

    # Decision variable 1: x[i, v, t] = 1 if AGV i is at node v at time t
    x = {}
    for i in range(n_agvs):
        for t in range(T + 1):
            for v_idx, v in enumerate(valid_nodes):
                x[i, v_idx, t] = model.addVar(vtype=GRB.BINARY,
                                              name=f"x_{i}_{v_idx}_{t}")

    # Decision variable 2: y[i, j] = 1 if AGV i executes task j
    y = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # Decision variable 3: C[i] = completion time of AGV i
    C = {}
    for i in range(n_agvs):
        C[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"C_{i}")

    # Decision variable 4: T_max = makespan
    T_max = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name="T_max")

    # Decision variable 5: t_pick[i,j] and t_drop[i,j] (pickup/dropoff start times)
    t_pick = {}
    t_drop = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            t_pick[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"t_pick_{i}_{j}")
            t_drop[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"t_drop_{i}_{j}")

    # Auxiliary variables: z_pick[i,j,t] = 1 if AGV i starts loading task j at time t
    z_pick = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                z_pick[i, j, t] = model.addVar(vtype=GRB.BINARY,
                                               name=f"z_pick_{i}_{j}_{t}")

    # Auxiliary variables: z_drop[i,j,t] = 1 if AGV i starts unloading task j at time t
    z_drop = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                z_drop[i, j, t] = model.addVar(vtype=GRB.BINARY,
                                               name=f"z_drop_{i}_{j}_{t}")

    model.update()

    print(f"  Variables added")
    print(f"  Adding constraints...")

    # ===== Constraints =====

    # 1. Initial position constraints
    for i in range(n_agvs):
        start_pos = tuple(agv_starts[i])
        start_v_idx = node_to_idx[start_pos]
        model.addConstr(x[i, start_v_idx, 0] == 1, f"init_{i}")

    # 2. Each AGV must be at exactly one location at each time step
    for i in range(n_agvs):
        for t in range(T + 1):
            model.addConstr(
                gp.quicksum(x[i, v_idx, t] for v_idx in range(n_nodes)) == 1,
                f"position_{i}_{t}"
            )

    # 3. Movement continuity: adjacent time steps must move to adjacent nodes or stay
    for i in range(n_agvs):
        for t in range(T):
            for v_idx, v in enumerate(valid_nodes):
                non_neighbors = []
                for u_idx, u in enumerate(valid_nodes):
                    dist_uv = abs(u[0] - v[0]) + abs(u[1] - v[1])
                    if dist_uv > 1:
                        non_neighbors.append(u_idx)

                for u_idx in non_neighbors:
                    model.addConstr(
                        x[i, u_idx, t] + x[i, v_idx, t + 1] <= 1,
                        f"move_{i}_{u_idx}_{v_idx}_{t}"
                    )

    # Note: Vertex and edge conflict constraints are excluded in this relaxed model

    # 4. Task assignment: each task must be assigned to exactly one AGV
    for j in range(n_tasks):
        model.addConstr(
            gp.quicksum(y[i, j] for i in range(n_agvs)) == 1,
            f"task_assign_{j}"
        )

    # 5. Each AGV executes at most one task
    for i in range(n_agvs):
        model.addConstr(
            gp.quicksum(y[i, j] for j in range(n_tasks)) <= 1,
            f"max_one_task_per_agv_{i}"
        )

    # 6. Task execution: if AGV i executes task j, it must visit pickup and dropoff
    for i in range(n_agvs):
        for j in range(n_tasks):
            pickup_pos = tuple(tasks[j]['pickup'])
            dropoff_pos = tuple(tasks[j]['dropoff'])
            pickup_idx_node = node_to_idx[pickup_pos]
            dropoff_idx_node = node_to_idx[dropoff_pos]

            model.addConstr(
                gp.quicksum(x[i, pickup_idx_node, t] for t in range(T + 1)) >= y[i, j],
                f"visit_pickup_{i}_{j}"
            )

            model.addConstr(
                gp.quicksum(x[i, dropoff_idx_node, t] for t in range(T + 1)) >= y[i, j],
                f"visit_dropoff_{i}_{j}"
            )

            # Continuous停留 constraints (following paper formulation)

            # Constraint 1: Exactly one loading start time
            model.addConstr(
                gp.quicksum(z_pick[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1))
                == y[i, j],
                f"one_loading_start_{i}_{j}"
            )

            # Constraint 2: Exactly one unloading start time
            model.addConstr(
                gp.quicksum(z_drop[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1))
                == y[i, j],
                f"one_unloading_start_{i}_{j}"
            )

            # Constraint 3: Must stay at pickup during loading
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                for s in range(LOAD_UNLOAD_TIME + 1):
                    if t + s <= T:
                        model.addConstr(
                            z_pick[i, j, t] <= x[i, pickup_idx_node, t + s] + (T + 1) * (1 - y[i, j]),
                            f"loading_stay_{i}_{j}_{t}_{s}"
                        )

            # Constraint 4: Must stay at dropoff during unloading
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                for s in range(LOAD_UNLOAD_TIME + 1):
                    if t + s <= T:
                        model.addConstr(
                            z_drop[i, j, t] <= x[i, dropoff_idx_node, t + s] + (T + 1) * (1 - y[i, j]),
                            f"unloading_stay_{i}_{j}_{t}_{s}"
                        )

            # Constraint 5: Define pickup start time
            model.addConstr(
                t_pick[i, j] == gp.quicksum(t * z_pick[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1)),
                f"def_t_pick_{i}_{j}"
            )

            # Constraint 6: Define dropoff start time
            model.addConstr(
                t_drop[i, j] == gp.quicksum(t * z_drop[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1)),
                f"def_t_drop_{i}_{j}"
            )

            # Temporal constraint: includes travel time
            pickup_pos_j = tuple(tasks[j]['pickup'])
            dropoff_pos_j = tuple(tasks[j]['dropoff'])
            dist_j = compute_shortest_path_distance(
                list(pickup_pos_j), list(dropoff_pos_j), obstacles, width, height
            )

            if dist_j != float('inf'):
                model.addConstr(
                    t_drop[i, j] >= t_pick[i, j] + LOAD_UNLOAD_TIME + dist_j - (T + 1) * (1 - y[i, j]),
                    f"order_{i}_{j}"
                )
            else:
                model.addConstr(y[i, j] == 0, f"infeasible_{i}_{j}")

    # 7. Completion time definition
    for i in range(n_agvs):
        for j in range(n_tasks):
            M = T + 1
            model.addConstr(
                C[i] >= t_drop[i, j] + LOAD_UNLOAD_TIME - M * (1 - y[i, j]),
                f"completion_{i}_{j}"
            )

    # 8. Makespan definition
    for i in range(n_agvs):
        model.addConstr(T_max >= C[i], f"makespan_{i}")

    # Objective: minimize makespan
    model.setObjective(T_max, GRB.MINIMIZE)

    model.update()
    print(f"  Constraints added")

    print(f"\nStep 2: Solving...")
    solve_start_time = time.time()
    model.optimize()
    solve_time = time.time() - solve_start_time

    print(f"\nStep 3: Results")
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print(f"  Status: {'Optimal' if model.Status == GRB.OPTIMAL else 'Time limit reached'}")

        # Extract task assignment
        assignment = {}
        for i in range(n_agvs):
            for j in range(n_tasks):
                if y[i, j].X > 0.5:
                    assignment[j] = i

        # Extract paths
        paths = {}
        for i in range(n_agvs):
            path = []
            for t in range(T + 1):
                for v_idx, v in enumerate(valid_nodes):
                    if x[i, v_idx, t].X > 0.5:
                        path.append({'t': t, 'x': v[0], 'y': v[1]})
                        break
            paths[i] = path

        makespan = int(T_max.X) if T_max.X is not None else float('inf')

        # Post-processing: ensure AGVs remain stationary after reaching dropoff
        print(f"\n  Post-processing: Ensuring AGVs stay at dropoff after arrival...")
        for agv_id in range(n_agvs):
            assigned_tasks = [tid for tid, aid in assignment.items() if aid == agv_id]
            if assigned_tasks:
                task_id = assigned_tasks[0]
                task = tasks[task_id]
                dropoff = tuple(task['dropoff'])

                first_dropoff_time = None
                for t in range(len(paths[agv_id])):
                    pos = (paths[agv_id][t]['x'], paths[agv_id][t]['y'])
                    if pos == dropoff:
                        first_dropoff_time = t
                        break

                if first_dropoff_time is not None:
                    for t2 in range(first_dropoff_time, len(paths[agv_id])):
                        paths[agv_id][t2]['x'] = dropoff[0]
                        paths[agv_id][t2]['y'] = dropoff[1]
                    print(f"    ✓ AGV{agv_id} (Task{task_id}): Stays at {dropoff} from t={first_dropoff_time}")
                else:
                    print(f"    ⚠ AGV{agv_id} (Task{task_id}): Dropoff {dropoff} not reached!")

        # Truncate paths to makespan length
        truncated_paths = {}
        for i, path in paths.items():
            truncated_paths[i] = path[:makespan + 1]

        print(f"\n  Makespan (lower bound): {makespan} steps")
        print(f"  Note: This is a relaxed solution; actual execution may require additional waiting time")
        print(f"\n  Assignment:")
        for j, i in assignment.items():
            print(f"    Task{j} → AGV{i}")

        if save_yaml:
            simple_paths = {}
            for i, path in truncated_paths.items():
                simple_paths[i] = [(p['x'], p['y']) for p in path]

            # Conflict detection
            print(f"\n  Verifying path conflicts...")
            conflicts = detect_conflicts(simple_paths)
            if conflicts:
                print(f"\n  ⚠ Warning: {len(conflicts)} conflicts detected!")
                print(f"  This solution is infeasible for complete MAPF but acceptable for relaxed model")
                print(f"  Actual execution will require additional waiting to avoid conflicts")
                for conflict in conflicts[:5]:
                    if conflict['type'] == 'vertex':
                        print(f"    - Vertex conflict: t={conflict['time']}, "
                              f"AGV{conflict['agvs'][0]} and AGV{conflict['agvs'][1]} at {conflict['position']}")
                    else:
                        print(f"    - Edge conflict: t={conflict['time']}, "
                              f"AGV{conflict['agvs'][0]} and AGV{conflict['agvs'][1]} "
                              f"{conflict['positions'][0]}↔{conflict['positions'][1]}")
            else:
                print(f"\n  ✓ No conflicts (relaxed solution is also feasible)")

            save_optimal_paths_yaml(simple_paths, makespan, output_file)

            return {
                'makespan': makespan,
                'assignment': assignment,
                'paths': truncated_paths,
                'runtime': solve_time,
                'optimal': model.Status == GRB.OPTIMAL,
                'has_conflicts': len(conflicts) > 0,
                'num_conflicts': len(conflicts),
                'is_lower_bound': True,
                'gap': model.MIPGap if hasattr(model, 'MIPGap') else None
            }

    else:
        print(f"\n  Solver failed (Status: {model.Status})")
        return None


# ============================================================
# Mode 2: Complete MIP (with conflict avoidance)
# ============================================================
def solve_mip_full_mapf(agv_starts, tasks, obstacles, width, height,
                        save_yaml=True, output_file='yaml/optimal_paths_full.yaml'):
    """
    Solve complete MIP with conflict avoidance.

    Strictly enforces:
    - Vertex conflict constraints (no two AGVs at same location)
    - Edge conflict constraints (no head-to-head collisions)
    """
    n_agvs = len(agv_starts)
    n_tasks = len(tasks)

    print("\n" + "=" * 60)
    print("【Mode 2】Complete MIP (With Conflict Avoidance)")
    print("=" * 60)

    # Estimate time horizon T
    dist_matrix, _, point_to_idx = compute_distance_matrix(
        agv_starts, tasks, obstacles, width, height
    )

    start_idx = [point_to_idx[tuple(p)] for p in agv_starts]
    pickup_idx = [point_to_idx[tuple(t['pickup'])] for t in tasks]
    dropoff_idx = [point_to_idx[tuple(t['dropoff'])] for t in tasks]

    max_dist = 0
    for k in range(n_agvs):
        for w in range(n_tasks):
            d = (dist_matrix[start_idx[k], pickup_idx[w]] +
                 LOAD_UNLOAD_TIME +
                 dist_matrix[pickup_idx[w], dropoff_idx[w]] +
                 LOAD_UNLOAD_TIME)
            if d != float('inf'):
                max_dist = max(max_dist, d)

    T = min(int(max_dist * 3) + 30, 150)
    print(f"\n  Estimated time horizon T = {T}")
    print(f"  Load/Unload Time: {LOAD_UNLOAD_TIME} step(s)")

    # Valid nodes
    valid_nodes = [(x, y) for x in range(width) for y in range(height)
                   if [x, y] not in obstacles]
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    n_nodes = len(valid_nodes)

    print(f"  Valid nodes: {n_nodes}")
    print(f"  Decision variables: ~{n_agvs * n_nodes * (T + 1):,}")

    print("\nStep 1: Building MIP model...")
    model = gp.Model("MAPF_Full_MIP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', TIME_LIMIT)
    model.setParam('Threads', 4)

    # Decision variables
    x = {}
    for i in range(n_agvs):
        for t in range(T + 1):
            for v_idx, v in enumerate(valid_nodes):
                x[i, v_idx, t] = model.addVar(vtype=GRB.BINARY,
                                              name=f"x_{i}_{v_idx}_{t}")

    y = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    C = {}
    for i in range(n_agvs):
        C[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"C_{i}")

    T_max = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name="T_max")

    t_pick = {}
    t_drop = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            t_pick[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"t_pick_{i}_{j}")
            t_drop[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"t_drop_{i}_{j}")

    z_pick = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                z_pick[i, j, t] = model.addVar(vtype=GRB.BINARY,
                                               name=f"z_pick_{i}_{j}_{t}")

    z_drop = {}
    for i in range(n_agvs):
        for j in range(n_tasks):
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                z_drop[i, j, t] = model.addVar(vtype=GRB.BINARY,
                                               name=f"z_drop_{i}_{j}_{t}")

    model.update()

    print(f"  Variables added")
    print(f"  Adding constraints...")

    # ===== Constraints =====

    # 1. Initial position constraints
    for i in range(n_agvs):
        start_pos = tuple(agv_starts[i])
        start_v_idx = node_to_idx[start_pos]
        model.addConstr(x[i, start_v_idx, 0] == 1, f"init_{i}")

    # 2. Position uniqueness
    for i in range(n_agvs):
        for t in range(T + 1):
            model.addConstr(
                gp.quicksum(x[i, v_idx, t] for v_idx in range(n_nodes)) == 1,
                f"position_{i}_{t}"
            )

    # 3. Movement continuity
    for i in range(n_agvs):
        for t in range(T):
            for v_idx, v in enumerate(valid_nodes):
                non_neighbors = []
                for u_idx, u in enumerate(valid_nodes):
                    dist_uv = abs(u[0] - v[0]) + abs(u[1] - v[1])
                    if dist_uv > 1:
                        non_neighbors.append(u_idx)

                for u_idx in non_neighbors:
                    model.addConstr(
                        x[i, u_idx, t] + x[i, v_idx, t + 1] <= 1,
                        f"move_{i}_{u_idx}_{v_idx}_{t}"
                    )

    # 4. Vertex conflict constraints
    for t in range(T + 1):
        for v_idx in range(n_nodes):
            model.addConstr(
                gp.quicksum(x[i, v_idx, t] for i in range(n_agvs)) <= 1,
                f"vertex_conflict_{v_idx}_{t}"
            )

    # 5. Edge conflict constraints
    for t in range(T):
        for u_idx, u in enumerate(valid_nodes):
            for v_idx, v in enumerate(valid_nodes):
                if u != v and abs(u[0] - v[0]) + abs(u[1] - v[1]) == 1:
                    for i in range(n_agvs):
                        for j in range(i + 1, n_agvs):
                            model.addConstr(
                                x[i, u_idx, t] + x[i, v_idx, t + 1] +
                                x[j, v_idx, t] + x[j, u_idx, t + 1] <= 3,
                                f"edge_conflict_{i}_{j}_{u_idx}_{v_idx}_{t}"
                            )

    # 6. Task assignment
    for j in range(n_tasks):
        model.addConstr(
            gp.quicksum(y[i, j] for i in range(n_agvs)) == 1,
            f"task_assign_{j}"
        )

    # 7. Each AGV at most one task
    for i in range(n_agvs):
        model.addConstr(
            gp.quicksum(y[i, j] for j in range(n_tasks)) <= 1,
            f"max_one_task_per_agv_{i}"
        )

    # 8. Task execution constraints
    for i in range(n_agvs):
        for j in range(n_tasks):
            pickup_pos = tuple(tasks[j]['pickup'])
            dropoff_pos = tuple(tasks[j]['dropoff'])
            pickup_idx_node = node_to_idx[pickup_pos]
            dropoff_idx_node = node_to_idx[dropoff_pos]

            model.addConstr(
                gp.quicksum(x[i, pickup_idx_node, t] for t in range(T + 1)) >= y[i, j],
                f"visit_pickup_{i}_{j}"
            )

            model.addConstr(
                gp.quicksum(x[i, dropoff_idx_node, t] for t in range(T + 1)) >= y[i, j],
                f"visit_dropoff_{i}_{j}"
            )

            # Continuous停留 constraints

            # Exactly one loading start time
            model.addConstr(
                gp.quicksum(z_pick[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1))
                == y[i, j],
                f"one_loading_start_{i}_{j}"
            )

            # Exactly one unloading start time
            model.addConstr(
                gp.quicksum(z_drop[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1))
                == y[i, j],
                f"one_unloading_start_{i}_{j}"
            )

            # Stay at pickup during loading
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                for s in range(LOAD_UNLOAD_TIME + 1):
                    if t + s <= T:
                        model.addConstr(
                            z_pick[i, j, t] <= x[i, pickup_idx_node, t + s] + (T + 1) * (1 - y[i, j]),
                            f"loading_stay_{i}_{j}_{t}_{s}"
                        )

            # Stay at dropoff during unloading
            for t in range(T - LOAD_UNLOAD_TIME + 1):
                for s in range(LOAD_UNLOAD_TIME + 1):
                    if t + s <= T:
                        model.addConstr(
                            z_drop[i, j, t] <= x[i, dropoff_idx_node, t + s] + (T + 1) * (1 - y[i, j]),
                            f"unloading_stay_{i}_{j}_{t}_{s}"
                        )

            # Define pickup start time
            model.addConstr(
                t_pick[i, j] == gp.quicksum(t * z_pick[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1)),
                f"def_t_pick_{i}_{j}"
            )

            # Define dropoff start time
            model.addConstr(
                t_drop[i, j] == gp.quicksum(t * z_drop[i, j, t] for t in range(T - LOAD_UNLOAD_TIME + 1)),
                f"def_t_drop_{i}_{j}"
            )

            # Temporal constraint with travel time
            pickup_pos_j = tuple(tasks[j]['pickup'])
            dropoff_pos_j = tuple(tasks[j]['dropoff'])
            dist_j = compute_shortest_path_distance(
                list(pickup_pos_j), list(dropoff_pos_j), obstacles, width, height
            )

            if dist_j != float('inf'):
                model.addConstr(
                    t_drop[i, j] >= t_pick[i, j] + LOAD_UNLOAD_TIME + dist_j - (T + 1) * (1 - y[i, j]),
                    f"order_{i}_{j}"
                )
            else:
                model.addConstr(y[i, j] == 0, f"infeasible_{i}_{j}")

    # 9. Completion time definition
    for i in range(n_agvs):
        for j in range(n_tasks):
            M = T + 1
            model.addConstr(
                C[i] >= t_drop[i, j] + LOAD_UNLOAD_TIME - M * (1 - y[i, j]),
                f"completion_{i}_{j}"
            )

    # 10. Makespan definition
    for i in range(n_agvs):
        model.addConstr(T_max >= C[i], f"makespan_{i}")

    # Objective: minimize makespan
    model.setObjective(T_max, GRB.MINIMIZE)

    model.update()
    print(f"  Constraints added")

    print(f"\nStep 2: Solving...")
    solve_start_time = time.time()
    model.optimize()
    solve_time = time.time() - solve_start_time

    print(f"\nStep 3: Results")
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print(f"  Status: {'Optimal' if model.Status == GRB.OPTIMAL else 'Time limit reached'}")

        # Extract assignment
        assignment = {}
        for i in range(n_agvs):
            for j in range(n_tasks):
                if y[i, j].X > 0.5:
                    assignment[j] = i

        # Extract paths
        paths = {}
        for i in range(n_agvs):
            path = []
            for t in range(T + 1):
                for v_idx, v in enumerate(valid_nodes):
                    if x[i, v_idx, t].X > 0.5:
                        path.append({'t': t, 'x': v[0], 'y': v[1]})
                        break
            paths[i] = path

        makespan = int(T_max.X) if T_max.X is not None else float('inf')

        # Post-processing: ensure AGVs remain at dropoff
        print(f"\n  Post-processing: Ensuring AGVs stay at dropoff after arrival...")
        for agv_id in range(n_agvs):
            assigned_tasks = [tid for tid, aid in assignment.items() if aid == agv_id]
            if assigned_tasks:
                task_id = assigned_tasks[0]
                task = tasks[task_id]
                dropoff = tuple(task['dropoff'])

                first_dropoff_time = None
                for t in range(len(paths[agv_id])):
                    pos = (paths[agv_id][t]['x'], paths[agv_id][t]['y'])
                    if pos == dropoff:
                        first_dropoff_time = t
                        break

                if first_dropoff_time is not None:
                    for t2 in range(first_dropoff_time, len(paths[agv_id])):
                        paths[agv_id][t2]['x'] = dropoff[0]
                        paths[agv_id][t2]['y'] = dropoff[1]
                    print(f"    ✓ AGV{agv_id} (Task{task_id}): Stays at {dropoff} from t={first_dropoff_time}")
                else:
                    print(f"    ⚠ AGV{agv_id} (Task{task_id}): Dropoff {dropoff} not reached!")

        # Truncate paths
        truncated_paths = {}
        for i, path in paths.items():
            truncated_paths[i] = path[:makespan + 1]

        print(f"\n  Makespan: {makespan} steps")
        print(f"\n  Assignment:")
        for j, i in assignment.items():
            print(f"    Task{j} → AGV{i}")

        if save_yaml:
            simple_paths = {}
            for i, path in truncated_paths.items():
                simple_paths[i] = [(p['x'], p['y']) for p in path]

            # Conflict verification
            print(f"\n  Verifying path conflicts...")
            conflicts = detect_conflicts(simple_paths)
            if conflicts:
                print(f"\n  ⚠ Warning: Complete MIP has {len(conflicts)} conflicts!")
                print(f"  This solution is actually infeasible")
                for conflict in conflicts[:5]:
                    if conflict['type'] == 'vertex':
                        print(f"    - Vertex conflict: t={conflict['time']}, "
                              f"AGV{conflict['agvs'][0]} and AGV{conflict['agvs'][1]} at {conflict['position']}")
                    else:
                        print(f"    - Edge conflict: t={conflict['time']}, "
                              f"AGV{conflict['agvs'][0]} and AGV{conflict['agvs'][1]} "
                              f"{conflict['positions'][0]}↔{conflict['positions'][1]}")
            else:
                print(f"\n  ✓ Complete MIP paths are conflict-free (truly feasible)")

            save_optimal_paths_yaml(simple_paths, makespan, output_file)

            return {
                'makespan': makespan,
                'assignment': assignment,
                'paths': truncated_paths,
                'runtime': solve_time,
                'optimal': model.Status == GRB.OPTIMAL,
                'gap': model.MIPGap if hasattr(model, 'MIPGap') else None
            }

    else:
        print(f"\n  Solver failed (Status: {model.Status})")
        return None


# ============================================================
# Main Program
# ============================================================
if __name__ == "__main__":
    # Load instance
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    # Load map
    width, height, obstacles = load_map_and_obstacles()

    print("=" * 60)
    print("Gurobi MIP Solver System")
    print("=" * 60)
    print(f"  Map: {MAP_FILE}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  AGVs: {len(instance['agv_starts'])}")
    print(f"  Tasks: {len(instance['tasks'])}")

    if MIP_MODE == 1:
        print(f"  Mode: Mode 1 (Relaxed MIP)")
        print(f"  Note: Conflict constraints removed")
    else:
        print(f"  Mode: Mode 2 (Complete MIP)")
        print(f"  Note: All constraints included")

    print(f"\n  AGV Start Positions: {instance['agv_starts']}")
    print(f"  Tasks:")
    for i, t in enumerate(instance['tasks']):
        print(f"    Task{i}: Pickup {t['pickup']} → Dropoff {t['dropoff']}")

    # Solve based on mode
    if MIP_MODE == 1:
        result = solve_mip_relaxed(
            instance['agv_starts'],
            instance['tasks'],
            obstacles,
            width, height,
            save_yaml=True,
            output_file='yaml/optimal_paths_relaxed.yaml'
        )
    elif MIP_MODE == 2:
        result = solve_mip_full_mapf(
            instance['agv_starts'],
            instance['tasks'],
            obstacles,
            width, height,
            save_yaml=True,
            output_file='yaml/optimal_paths_full.yaml'
        )
    else:
        raise ValueError(f"Invalid mode: {MIP_MODE}, choose 1 or 2")

    if result and result.get('optimal'):
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        print(f"  Optimal Makespan: {result['makespan']:.0f} steps")
        print(f"  Assignment: {result['assignment']}")
        print(f"\n Gurobi Computation Time: {result['runtime']:.4f} seconds")

        if MIP_MODE == 1:
            print(f"\n  【Relaxed MIP Notes】")
            print(f"  This model is a relaxation of complete MIP")
            print(f"  Removed constraints: vertex conflicts, edge conflicts")
            print(f"  Theoretical guarantee: Makespan_relaxed ≤ Makespan_complete")
            if result.get('has_conflicts'):
                print(f"\n  ⚠ Detected {result.get('num_conflicts', 0)} conflicts")
                print(f"  This solution is infeasible for complete MAPF but acceptable as lower bound")
            else:
                print(f"\n  ✓ Solution is conflict-free, also feasible for complete MAPF")
        else:
            print(f"\n  【Complete MIP Notes】")
            print(f"  Solution strictly satisfies all constraints including conflict avoidance")
            print(f"  This is a truly optimal and feasible solution")
    else:
        print("\n  ⚠ No optimal solution found")

