import yaml
import random
import networkx as nx
import copy
import matplotlib.pyplot as plt
from PIL import Image
import time
from decoder import Decoder, Decoder_Incremental


def load_map_agv_task(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    grid = data['map']['data']
    height = data['map']['height']
    width = data['map']['width']

    G = nx.grid_2d_graph(height, width)
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                if (y, x) in G:
                    G.remove_node((y, x))

    agvs = [tuple(agv['position']) for agv in data['agvs']]

    initial_tasks = [{'pickup': tuple(t['pickup']), 
                      'dropoff': tuple(t['dropoff'])}
                     for t in data['tasks'].get('initial', [])]

    delayed_tasks = [
        {
            'step': d['step'],
            'pickup': tuple(d['task']['pickup']),
            'dropoff': tuple(d['task']['dropoff'])
        }
        for d in data['tasks'].get('delayed', [])
    ]

    return G, agvs, initial_tasks, delayed_tasks, grid


def calc_astar_cost(G, start, goal, cache=None):
    key = (start, goal)
    if cache is not None and key in cache:
        return cache[key]
    try:
        if start not in G or goal not in G:
            result = float('inf')
        else:
            path = nx.astar_path(G, start, goal)
            result = len(path) - 1
    except nx.NetworkXNoPath:
        result = float('inf')
    except Exception:
        result = float('inf')
    if cache is not None:
        cache[key] = result
    return result


def init_population(num_agv, num_tasks, pop_size):
    population = []
    for _ in range(pop_size):
        if num_tasks >= num_agv:
            task_list = list(range(num_tasks))
            random.shuffle(task_list)
            indiv = task_list[:num_agv]
        else:
            task_list = list(range(num_tasks))
            virtual_tasks = [-1 - i for i in range(num_agv - num_tasks)]
            all_tasks = task_list + virtual_tasks
            random.shuffle(all_tasks)
            indiv = all_tasks
        population.append(indiv)
    return population


def crossover(p1, p2):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    n = len(p1)
    idx1 = random.randint(0, n - 1)
    idx2 = random.randint(0, n - 1)

    val_from_p2 = p2[idx2]
    val_from_p1 = p1[idx1]

    c1[idx1] = val_from_p2
    c2[idx2] = val_from_p1

    dup_c1 = None
    dup_c2 = None
    for i in range(n):
        if i != idx1 and c1[i] == val_from_p2:
            dup_c1 = i
            break
    for i in range(n):
        if i != idx2 and c2[i] == val_from_p1:
            dup_c2 = i
            break

    if dup_c1 is not None and dup_c2 is not None:
        c1[dup_c1], c2[dup_c2] = c2[dup_c2], c1[dup_c1]

    return c1, c2


def mutate(indiv):
    if len(indiv) < 2:
        return
    idx1, idx2 = random.sample(range(len(indiv)), 2)
    indiv[idx1], indiv[idx2] = indiv[idx2], indiv[idx1]


def evaluate_with_completion_time(individual, tasks, agv_states, G, cache=None):
    completion_times = []
    total_completion = 0

    for agv_id, task_id in enumerate(individual):
        if task_id < 0 or task_id >= len(tasks):
            completion_times.append(0)
            continue

        task = tasks[task_id]
        current_pos = agv_states[agv_id]['position']

        dist_to_pickup = calc_astar_cost(G, current_pos, task['pickup'], cache)
        dist_to_dropoff = calc_astar_cost(G, task['pickup'], task['dropoff'], cache)
        completion_time = dist_to_pickup + dist_to_dropoff + 2
        completion_times.append(completion_time)
        total_completion += completion_time

    return max(completion_times) + total_completion / 1000 if completion_times else 0


def genetic_algorithm(G, tasks, agv_states, generations=50, pop_size=50, mutation_rate=0.5):
    num_agv = len(agv_states)
    num_tasks = len(tasks)

    print(f"  [GA内部確認] 世代数: {generations}, 個体数: {pop_size}, 変異率: {mutation_rate}")

    cache = {}  # 本次GA运行内的A*距离缓存（G固定，键为(start,goal)）
    population = init_population(num_agv, num_tasks, pop_size)

    for gen in range(generations):
        population.sort(key=lambda ind: evaluate_with_completion_time(ind, tasks, agv_states, G, cache))

        elite_size = pop_size // 2
        next_gen = copy.deepcopy(population[:elite_size])

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:elite_size], 2)
            p1, p2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1, c2 = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(c1)
            if random.random() < mutation_rate:
                mutate(c2)
            next_gen.extend([c1, c2])

        population = copy.deepcopy(next_gen[:pop_size])

    best = min(population, key=lambda ind: evaluate_with_completion_time(ind, tasks, agv_states, G, cache))
    best_cost = evaluate_with_completion_time(best, tasks, agv_states, G, cache)

    result = [[] for _ in range(num_agv)]
    for agv_id, task_id in enumerate(best):
        if task_id >= 0:
            result[agv_id].append(task_id)

    return result, best_cost


def evaluate_delayed_tasks_with_completion(individual, tasks, agv_states, G, completion_info):
    total_cost = 0
    total_completion = 0
    
    for agv_id, task_id in enumerate(individual):
        if task_id < 0 or task_id >= len(tasks):
            continue
            
        task = tasks[task_id]
        info = completion_info[agv_id]
        start_pos = info['completion_position']
        completion_step = info['completion_step']
        
        dist_to_pickup = calc_astar_cost(G, start_pos, task['pickup'])
        dist_to_dropoff = calc_astar_cost(G, task['pickup'], task['dropoff'])
        task_completion_time = completion_step + dist_to_pickup + dist_to_dropoff + 2
        total_completion += task_completion_time
        if total_cost < task_completion_time:
            total_cost = task_completion_time
    
    return total_cost + total_completion / 1000


def genetic_algorithm_delayed_task(G, tasks, agv_states, completion_info, 
                                     generations=50, pop_size=50, mutation_rate=0.5):
    num_agv = len(agv_states)
    num_tasks = len(tasks)
    
    population = init_population(num_agv, num_tasks, pop_size)
    
    for gen in range(generations):
        population.sort(key=lambda ind: evaluate_delayed_tasks_with_completion(ind, tasks, agv_states, G, completion_info))
        
        elite_size = pop_size // 2
        next_gen = copy.deepcopy(population[:elite_size])
        
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:elite_size], 2)
            p1, p2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1, c2 = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(c1)
            if random.random() < mutation_rate:
                mutate(c2)
            next_gen.extend([c1, c2])
        
        population = copy.deepcopy(next_gen[:pop_size])
    
    best = min(population, key=lambda ind: evaluate_delayed_tasks_with_completion(ind, tasks, agv_states, G, completion_info))
    best_cost = evaluate_delayed_tasks_with_completion(best, tasks, agv_states, G, completion_info)
    
    result = [[] for _ in range(num_agv)]
    for agv_id, task_id in enumerate(best):
        if task_id >= 0:
            result[agv_id].append(task_id)
    
    return result, best_cost


def validate_and_fix_tasks(tasks, G, grid):
    fixed_tasks = []
    for i, task in enumerate(tasks):
        pickup = task['pickup']
        dropoff = task['dropoff']
        if pickup not in G:
            pickup = find_nearest_valid_node(pickup, G)
        if dropoff not in G:
            dropoff = find_nearest_valid_node(dropoff, G)
        fixed_tasks.append({'pickup': pickup, 'dropoff': dropoff})
    return fixed_tasks


def find_nearest_valid_node(target, G):
    min_dist = float('inf')
    nearest = None
    ty, tx = target
    for node in G.nodes():
        ny, nx_val = node
        dist = abs(ty - ny) + abs(tx - nx_val)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


def load_path(yaml_path='yaml/ga_output.yaml'):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None or 'schedule' not in data:
            return None
        q_paths = {}
        for agent_key, path_data in data['schedule'].items():
            agent_id = int(agent_key.replace('agent', ''))
            q_paths[agent_id] = [(step['y'], step['x']) for step in path_data]
        return q_paths
    except Exception:
        return None


def get_completion_info_from_ga_output(current_step, initial_tasks, task_queues, G, 
                                       yaml_path='yaml/ga_output.yaml'):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    
    schedule = data.get('schedule', {})
    completion_info = {}
    
    for agent_key, path_data in schedule.items():
        agv_id = int(agent_key.replace('agent', ''))
        if not path_data:
            continue
        
        current_pos = None
        for step_data in path_data:
            if step_data['t'] == current_step:
                current_pos = (step_data['y'], step_data['x'])
                break
        if current_pos is None:
            last_step = path_data[-1]
            current_pos = (last_step['y'], last_step['x'])
        
        agv_initial_task = None
        if task_queues and agv_id < len(task_queues) and task_queues[agv_id]:
            agv_initial_task = task_queues[agv_id][0]
        
        if agv_initial_task:
            pickup_pos = agv_initial_task['pickup']
            dropoff_pos = agv_initial_task['dropoff']
            
            pickup_visited = False
            dropoff_arrival_step = None
            stop_time = 1
            for step_data in path_data:
                if step_data['t'] > current_step:
                    break
                if (step_data['y'], step_data['x']) == pickup_pos:
                    pickup_visited = True
                if pickup_visited and (step_data['y'], step_data['x']) == dropoff_pos:
                    dropoff_arrival_step = step_data['t']

            if current_pos == dropoff_pos and dropoff_arrival_step is not None:
                if current_step >= dropoff_arrival_step + stop_time:
                    remaining_steps = 0
                else:
                    remaining_steps = (dropoff_arrival_step + stop_time) - current_step
            elif pickup_visited:
                dist_to_dropoff = calc_astar_cost(G, current_pos, dropoff_pos)
                if dist_to_dropoff == float('inf'):
                    dist_to_dropoff = 100
                remaining_steps = dist_to_dropoff + 1
            else:
                dist_to_pickup = calc_astar_cost(G, current_pos, pickup_pos)
                dist_pickup_to_dropoff = calc_astar_cost(G, pickup_pos, dropoff_pos)
                if dist_to_pickup == float('inf'):
                    dist_to_pickup = 100
                if dist_pickup_to_dropoff == float('inf'):
                    dist_pickup_to_dropoff = 100
                remaining_steps = dist_to_pickup + 1 + dist_pickup_to_dropoff + 1
            
            completion_info[agv_id] = {
                'completion_step': remaining_steps,
                'completion_position': dropoff_pos,
                'pickup_position': pickup_pos,
                'status': 'executing_initial_task'
            }
        else:
            completion_info[agv_id] = {
                'completion_step': 0,
                'completion_position': current_pos,
                'status': 'idle'
            }
    
    return completion_info


def assign_delayed_tasks_only(agv_states, new_tasks, G, current_step=0, 
                          initial_tasks=None, task_queues=None):
    completion_info = get_completion_info_from_ga_output(
        current_step, initial_tasks, task_queues, G
    )
    if completion_info is None:
        completion_info = {}
        for i, agv in enumerate(agv_states):
            completion_info[i] = {
                'completion_step': 0,
                'completion_position': agv['position']
            }
    
    best_assignment, best_cost = genetic_algorithm_delayed_task(
        G, new_tasks, agv_states, completion_info
    )
    
    real_task_assignment = [[] for _ in agv_states]
    for agv_id, task_indices in enumerate(best_assignment):
        for task_idx in task_indices:
            if task_idx >= 0:
                real_task_assignment[agv_id].append(new_tasks[task_idx])
    
    return real_task_assignment


def visualize_current_state(G, grid, agv_positions, all_tasks, step, save_path=None):
    height = len(grid)
    width = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    for i in range(width + 1):
        ax.axvline(x=i, color='gray', linewidth=0.5)
    for i in range(height + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5)
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, 
                                         facecolor='black', edgecolor='black'))
    
    task_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    task_positions = {}
    
    for task_id, task in enumerate(all_tasks):
        py, px = task['pickup']
        if (px, py) not in task_positions:
            task_positions[(px, py)] = []
        task_positions[(px, py)].append((task_colors[task_id % len(task_colors)], task_id, 'P'))
        
        dy, dx = task['dropoff']
        if (dx, dy) not in task_positions:
            task_positions[(dx, dy)] = []
        task_positions[(dx, dy)].append((task_colors[task_id % len(task_colors)], task_id, 'D'))
    
    for (x, y), task_list in task_positions.items():
        if len(task_list) == 1:
            color, task_id, task_type = task_list[0]
            ax.add_patch(plt.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, 
                                     facecolor=color, edgecolor='black', linewidth=1))
            ax.text(x + 0.5, y + 0.5, f'{task_id+1}{task_type}', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        else:
            for i, (color, task_id, task_type) in enumerate(task_list):
                w, h = 0.3, 0.3
                offset_x = 0.2 + (i % 2) * 0.3
                offset_y = 0.2 + (i // 2) * 0.3
                ax.add_patch(plt.Rectangle((x + offset_x, y + offset_y), w, h, 
                                         facecolor=color, edgecolor='black', linewidth=0.5))
                ax.text(x + offset_x + w/2, y + offset_y + h/2, f'{task_id+1}{task_type}', 
                       ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    for agv_id, pos in agv_positions.items():
        y, x = pos
        circle = plt.Circle((x + 0.5, y + 0.5), 0.3, 
                          facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x + 0.5, y + 0.5, str(agv_id + 1), 
               ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_title(f'Step {step}', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_simulation_gif(G, grid, agvs, initial_tasks, delayed_tasks, gif_path='agv_simulation.gif'):
    try:
        with open('yaml/ga_output.yaml', 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        q_learning_paths = {}
        for agent_key, path_data in data['schedule'].items():
            agent_id = int(agent_key.replace('agent', ''))
            q_learning_paths[agent_id] = [(step['y'], step['x']) for step in path_data]
    except Exception as e:
        return
    
    all_tasks = initial_tasks[:]
    for task in delayed_tasks:
        all_tasks.append({'pickup': task['pickup'], 'dropoff': task['dropoff']})
    
    max_steps = max(len(path) for path in q_learning_paths.values())
    
    frames = []
    for step in range(max_steps):
        agv_positions = {}
        for agv_id, path in q_learning_paths.items():
            agv_positions[agv_id] = path[step] if step < len(path) else path[-1]
        
        temp_path = f'temp_frame_{step:03d}.png'
        visualize_current_state(G, grid, agv_positions, all_tasks, step, temp_path)
        img = Image.open(temp_path)
        frames.append(img.copy())
        img.close()
    
    for i in range(3):
        temp_path = f'temp_frame_{max_steps+i:03d}.png'
        visualize_current_state(G, grid, agv_positions, all_tasks, max_steps - 1, temp_path)
        img = Image.open(temp_path)
        frames.append(img.copy())
        img.close()
    
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

    import os
    for idx in range(len(frames)):
        temp_path = f'temp_frame_{idx:03d}.png'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    t1 = time.time()
    yaml_path = "map_normal_congestion_low_compare.yaml"

    G, agvs, initial_tasks, delayed_tasks, grid = load_map_agv_task(yaml_path)
    initial_tasks = validate_and_fix_tasks(initial_tasks, G, grid)
    
    agv_states = [{'agv_id': i, 'position': pos} for i, pos in enumerate(agvs)]
    pending_tasks = initial_tasks[:]

    delay_task_dict = {}
    for task in delayed_tasks:
        step = task['step']
        delay_task_dict.setdefault(step, []).append({
            'pickup': task['pickup'],
            'dropoff': task['dropoff']
        })

    # 初期タスク
    best, predicted_steps = genetic_algorithm(G, pending_tasks, agv_states)
    
    dec = Decoder(best)
    initial_step_count = dec.main()
    
    task_queues = [[] for _ in agv_states]
    for agv_id, task_ids in enumerate(best):
        task_queues[agv_id] = [pending_tasks[t_id] for t_id in task_ids]

    # 追加タスク
    delayed_task_assignments = {}
    
    if delay_task_dict:
        for delay_step, new_tasks in sorted(delay_task_dict.items()):
            existing_paths = load_path('yaml/ga_output.yaml')
            if existing_paths is None:
                continue
            
            completion_info = get_completion_info_from_ga_output(
                delay_step, initial_tasks, task_queues, G
            )
            if completion_info is None:
                continue
            
            for agv_id in completion_info:
                completion_info[agv_id]['delay_step'] = delay_step
                completion_info[agv_id]['current_step'] = delay_step
                completion_info[agv_id]['absolute_completion_step'] = (
                    delay_step + completion_info[agv_id]['completion_step']
                )
            
            new_task_assignment = assign_delayed_tasks_only(
                agv_states, new_tasks, G,
                current_step=delay_step,
                initial_tasks=initial_tasks,
                task_queues=task_queues
            )
            delayed_task_assignments[delay_step] = new_task_assignment

            dec_inc = Decoder_Incremental(
                new_task_assignment=new_task_assignment,
                new_tasks=new_tasks,
                existing_paths=existing_paths,
                agv_completion_info=completion_info
            )
            
            try:
                step_qlearning = dec_inc.main()
                final_step_count = step_qlearning
                
                for agv_id, new_agv_tasks in enumerate(new_task_assignment):
                    task_queues[agv_id].extend(new_agv_tasks)
                pending_tasks.extend(new_tasks)
            except Exception:
                pass
    
    t2 = time.time()
    
    print("\nInitial task assignment:")
    for agv_id, task_ids in enumerate(best):
        if task_ids:
            print(f"  AGV{agv_id}: {[initial_tasks[t_id] for t_id in task_ids]}")

    if delayed_task_assignments:
        print("\nDelayed task assignment:")
        for delay_step, assignment in sorted(delayed_task_assignments.items()):
            print(f"  Step {delay_step}:")
            for agv_id, tasks in enumerate(assignment):
                if tasks:
                    print(f"    AGV{agv_id}: {tasks}")
    
    print(f"\nTotal steps: {final_step_count - 1}")
    print(f"Computation time: {t2 - t1} sec")
    
    # GIF
    create_simulation_gif(G, grid, agvs, initial_tasks, delayed_tasks, 'agv_simulation.gif')