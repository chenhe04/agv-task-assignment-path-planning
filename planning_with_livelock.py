# planning_with_livelock.py
"""
集成活锁检测的多AGV路径规划模块（方案C版）

改进内容：
1. 方案B：Phase 2 开头检查已到达目标的agent，跳过Q-learning（line 247-265）
2. 方案C：get_state() 中检测其他AGV是否已完成，跳过已完成的冲突源（line 155-171）
3. Phase 2.5：活锁检测与解除
4. 动态缓冲区：活锁解除后短期增大缓冲区防止索引越界

核心改动仅在 get_state() 和 main() 中，其余逻辑与 planning.py 完全一致。
"""

import numpy as np
import yaml
import time
import os
from astar_class import AStar
from livelock_detector import LivelockDetector, LivelockType

# ============================================================================
# Constant Definitions（与 planning.py 完全一致）
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

ACTION_SET = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]])

OBSERVATION_NODES = np.array([
    [0, 2], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1],
    [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [4, 2]
])

OBSTACLE_NODES = np.array([[1, 2], [2, 1], [2, 3], [3, 2]])

BASE_STOP_TIME = 1
LOAD_UNLOAD_TIME = 1
MAX_PLANNING_STEPS = 300

# ========== 活锁检测参数（新增） ==========
LIVELOCK_WINDOW_SIZE = 12
LIVELOCK_OSCILLATION_THRESHOLD = 10
LIVELOCK_MAX_COUNT = 20


class PlanningWithLivelock:
    """
    集成活锁检测的多AGV路径规划器（方案C版）。

    与原始 Planning 类的区别：
    1. get_state() 中检测已完成的AGV，跳过冲突检测（方案C）
    2. main() 中 Phase 2 开头检查已到达目标的agent，跳过Q-learning（方案B）
    3. Phase 2.5 活锁检测与解除
    4. 动态缓冲区机制
    """

    def __init__(self, agv_starts=None, tasks=None, output_file='yaml/ga_output.yaml'):
        self.output_file = output_file

        with open(_MAP_FILE, 'r', encoding='utf-8') as f:
            map_data = yaml.load(f, Loader=yaml.Loader)

        self.col = map_data['map']['width']
        self.row = map_data['map']['height']
        self.agent_size = len(agv_starts)

        self.obs = []
        for y in range(self.row):
            for x in range(self.col):
                if map_data['map']['data'][y][x] == 1:
                    self.obs.append([x, y])

        self.task_pos = np.zeros(self.agent_size, dtype=int)
        self.pick_or_place = np.full(self.agent_size, 0, dtype=int)

        self.task_list = {}
        self.agent_pos = {}

        for agent in range(self.agent_size):
            initial_pos = agv_starts[agent]
            self.task_list[str(agent)] = [initial_pos]
            self.agent_pos[str(agent)] = [initial_pos]

        self.custom_tasks = tasks

        for agent_idx in range(min(len(tasks), self.agent_size)):
            task = tasks[agent_idx]
            self.task_list[str(agent_idx)].append(task['pickup'])
            self.task_list[str(agent_idx)].append(task['dropoff'])

        self.state = np.full((self.col, self.row), -2, dtype=int)
        for obstacle in self.obs:
            self.state[obstacle[0], obstacle[1]] = -1

        for _ in range(2):
            self.state = np.insert(self.state, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = np.insert(self.state.T, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = self.state.T

        self.q_function = np.load('yaml/qfunction.npy')

        # ========== 活锁检测器（新增） ==========
        self.livelock_detector = LivelockDetector(
            num_agents=self.agent_size,
            window_size=LIVELOCK_WINDOW_SIZE,
            oscillation_threshold=LIVELOCK_OSCILLATION_THRESHOLD,
            max_livelocks=LIVELOCK_MAX_COUNT
        )

    def astar(self, t, agent, goal, start):
        a_star = AStar(self.col, self.row, goal, start, False, self.obs)
        path = a_star.main()

        if path:
            for node in path:
                self.agent_pos[str(agent)].append([node.x, node.y])

            self.agent_pos[str(agent)].append([path[-1].x, path[-1].y])

            return True
        return False

    def get_state(self, t, agent):
        passing = np.zeros(len(ACTION_SET) - 1, dtype=int)
        state_ = np.copy(self.state)

        a = np.array(self.agent_pos[str(agent)][t + 1]) - \
            np.array(self.agent_pos[str(agent)][t])
        idx = 0
        for i in range(len(ACTION_SET)):
            if np.all(ACTION_SET[i] - a == 0):
                idx = i
                break

        for i in range(self.agent_size):
            if i != agent:
                # ========== 方案C：改进冲突检测逻辑 ==========
                # 检查其他AGV是否已经到达最终目标并停下
                # 如果已到达，不将其视为冲突源
                is_completed = False
                if (self.task_pos[i] == len(self.task_list[str(i)]) - 1 and
                        t + BASE_STOP_TIME < len(self.agent_pos[str(i)])):
                    final_goal = np.array(self.task_list[str(i)][-1])
                    current_pos_i = np.array(self.agent_pos[str(i)][t + BASE_STOP_TIME])
                    if np.all(current_pos_i == final_goal):
                        is_completed = True

                # 如果其他AGV已完成任务并停下，跳过冲突检测
                if is_completed:
                    continue
                # ====================================================

                for j in range(len(ACTION_SET) - 1):
                    temp = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[j]
                    if (np.all(np.array(self.agent_pos[str(i)][t]) - temp == 0) and
                            np.all(np.array(self.agent_pos[str(i)][t + 1]) -
                                   np.array(self.agent_pos[str(agent)][t]) == 0)):
                        passing[j] = 1
                state_[self.agent_pos[str(i)][t + 1][0] + 2,
                       self.agent_pos[str(i)][t + 1][1] + 2] = 0

        local_state = state_[
            self.agent_pos[str(agent)][t][0]:self.agent_pos[str(agent)][t][0] + 5,
            self.agent_pos[str(agent)][t][1]:self.agent_pos[str(agent)][t][1] + 5
        ]

        for _ in range(idx):
            local_state = np.rot90(local_state, -1)

        return local_state, idx, passing

    def _resolve_livelock(self, t, livelock_agents):
        """
        解除活锁：打破对称性策略

        将涉及的AGV按ID排序，前半部分强制停留2步，后半部分重新规划。
        """
        print(f"    [LIVELOCK #{self.livelock_detector.livelock_count + 1}] "
              f"t={t}, 涉及AGV: {livelock_agents}")

        priority_sorted = sorted(livelock_agents)
        wait_count = max(1, len(priority_sorted) // 2)
        wait_agents = priority_sorted[:wait_count]
        move_agents = priority_sorted[wait_count:]

        print(f"      解除策略: 打破对称性")
        print(f"      等待组: {wait_agents} (强制停留2步)")
        print(f"      移动组: {move_agents} (重新规划)")

        for agent in wait_agents:
            current_pos = self.agent_pos[str(agent)][t]
            for _ in range(2 + BASE_STOP_TIME):
                self.agent_pos[str(agent)].append(current_pos.copy())

        for agent in move_agents:
            current_pos = self.agent_pos[str(agent)][t]
            goal = self.task_list[str(agent)][self.task_pos[agent]]

            self.agent_pos[str(agent)] = self.agent_pos[str(agent)][:t + 1]
            self.astar(t, agent, goal, current_pos)

        return True

    def main(self):
        t = 0
        start_time = time.time()

        # ========== 活锁解除后的缓冲区标志（新增） ==========
        livelock_just_resolved = False
        buffer_duration = 0

        while True:
            # ========================================================================
            # Phase 1: Task Assignment and A* Path Planning
            # ========================================================================
            for agent in range(self.agent_size):
                # 【改动】动态缓冲区：活锁解除后短期使用更大的缓冲区
                if livelock_just_resolved and buffer_duration > 0:
                    buffer_size = max(BASE_STOP_TIME, 3) + 1
                else:
                    buffer_size = BASE_STOP_TIME

                while len(self.agent_pos[str(agent)]) <= t + buffer_size:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])
                target_pos = np.array(self.task_list[str(agent)][self.task_pos[agent]])

                if np.all(current_pos - target_pos == 0):
                    if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                        next_target = self.task_list[str(agent)][self.task_pos[agent] + 1]
                        start_pos = self.agent_pos[str(agent)][t + BASE_STOP_TIME]

                        success = self.astar(t, agent, next_target, start_pos)

                        if success:
                            self.task_pos[agent] += 1
                    else:
                        self.agent_pos[str(agent)].append(
                            self.agent_pos[str(agent)][t + BASE_STOP_TIME]
                        )

            # ========================================================================
            # Phase 2: Q-learning-based Collision Avoidance
            # ========================================================================
            for agent in range(self.agent_size):
                # 【方案B】检查agent是否处于关键停留点
                # 关键停留点：已到达任务目标点（pickup或dropoff），需要停留装卸货
                # 如果处于关键停留点，跳过Q-learning避障，防止被推离
                if t + BASE_STOP_TIME < len(self.agent_pos[str(agent)]):
                    current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])

                    # 检查是否到达当前任务目标点
                    if self.task_pos[agent] < len(self.task_list[str(agent)]):
                        target_pos = np.array(self.task_list[str(agent)][self.task_pos[agent]])

                        # 如果当前位置 == 当前目标位置，说明正在停留装卸货
                        # 跳过Q-learning，保持原地不动
                        if np.all(current_pos == target_pos):
                            while len(self.agent_pos[str(agent)]) <= t + 1:
                                self.agent_pos[str(agent)].append(
                                    self.agent_pos[str(agent)][-1]
                                )
                            continue  # 跳过本agent的Q-learning

                # 以下为原始Q-learning逻辑（未到达目标点的agent）
                while len(self.agent_pos[str(agent)]) <= t + 1:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                local_state, rotate, passing = self.get_state(t, agent)

                goal = np.array(self.task_list[str(agent)][self.task_pos[agent]]) - \
                       np.array(self.agent_pos[str(agent)][t])

                if abs(goal[0]) >= abs(goal[1]):
                    muki = 0 if goal[0] < 0 else 2
                else:
                    muki = 1 if goal[1] < 0 else 3
                muki = (muki - rotate) % 4

                agv_node = '0000000000000'
                obs_node = '0000'
                passing_ = np.zeros(len(ACTION_SET) - 1, dtype=int)

                for i in range(len(OBSERVATION_NODES)):
                    if local_state[OBSERVATION_NODES[i][0]][OBSERVATION_NODES[i][1]] == 0:
                        agv_node = agv_node[:i] + '1' + agv_node[i + 1:]

                for i in range(len(OBSTACLE_NODES)):
                    if local_state[OBSTACLE_NODES[i][0]][OBSTACLE_NODES[i][1]] == -1:
                        obs_node = obs_node[:i] + '1' + obs_node[i + 1:]

                goal_node = format(muki, '02b')
                for p in range(len(passing)):
                    if passing[p] == 1:
                        passing_[(4 + p - rotate) % len(passing)] = 1

                passing__ = passing_[0] * 8 + passing_[1] * 4 + \
                            passing_[2] * 2 + passing_[3]
                passing__ = format(passing__, '04b')
                state_number = int(agv_node + obs_node + goal_node + passing__, 2)

                q = self.q_function[state_number]
                max_value = np.max(q)
                max_index = [a for a in range(len(q)) if q[a] == max_value]
                action = max_index[np.random.randint(0, len(max_index))]

                if (action != 0 and
                        not np.all(np.array(self.agent_pos[str(agent)][t]) -
                                   np.array(self.agent_pos[str(agent)][t + 1]) == 0)):
                    if action != 4:
                        action = (action + rotate) % 4

                    self.agent_pos[str(agent)] = self.agent_pos[str(agent)][:t + 1]
                    new_pos = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[action]
                    self.agent_pos[str(agent)].append(new_pos.tolist())

                    self.astar(t, agent,
                               self.task_list[str(agent)][self.task_pos[agent]],
                               new_pos.tolist())

            # ========================================================================
            # Phase 2.5: 活锁检测与解除（新增）
            # ========================================================================
            for agent in range(self.agent_size):
                if t < len(self.agent_pos[str(agent)]):
                    self.livelock_detector.record_position(
                        agent, self.agent_pos[str(agent)][t], t
                    )

            is_livelock, livelock_agents, livelock_type = \
                self.livelock_detector.detect_livelock(t)

            if is_livelock:
                if self.livelock_detector.should_stop():
                    elapsed_time = time.time() - start_time
                    print(f"\n[ERROR] 活锁次数过多 ({self.livelock_detector.livelock_count})，规划失败")
                    self.livelock_detector.print_summary()
                    return 0, elapsed_time, False

                success = self._resolve_livelock(t, livelock_agents)

                self.livelock_detector.log_livelock(
                    timestep=t,
                    agents=livelock_agents,
                    livelock_type=livelock_type,
                    success=success,
                    method="打破对称性"
                )

                if not success:
                    elapsed_time = time.time() - start_time
                    print(f"[ERROR] 活锁解除失败")
                    return 0, elapsed_time, False

                self.livelock_detector.reset_history(livelock_agents, keep_last=3)

                livelock_just_resolved = True
                buffer_duration = 3
            else:
                if buffer_duration > 0:
                    buffer_duration -= 1
                    if buffer_duration == 0:
                        livelock_just_resolved = False
            # ========================================================================

            t += 1

            # ========================================================================
            # Phase 3: Termination Condition Check
            # ========================================================================
            all_completed = True
            for agent in range(self.agent_size):
                if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                    all_completed = False
                    break

                # 【改动】安全访问：动态缓冲区下可能索引越界
                check_idx = min(t + BASE_STOP_TIME, len(self.agent_pos[str(agent)]) - 1)
                current_pos = np.array(self.agent_pos[str(agent)][check_idx])
                final_goal = np.array(self.task_list[str(agent)][-1])
                if not np.all(current_pos == final_goal):
                    all_completed = False
                    break

            if all_completed:
                elapsed_time = time.time() - start_time

                # 打印活锁统计（新增）
                self.livelock_detector.print_summary()

                # ====================================================================
                # Phase 4: Solution Validation (Post-verification)
                # ====================================================================
                all_tasks_valid = True

                for task_idx, task in enumerate(self.custom_tasks):
                    pickup = tuple(task['pickup'])
                    dropoff = tuple(task['dropoff'])

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

                if not all_tasks_valid:
                    print("[ERROR] Incomplete tasks detected, returning 0 steps "
                          "(infeasible solution)")
                    return 0, elapsed_time, False

                all_have_initial_wait = all(
                    len(self.agent_pos[str(a)]) >= 2 and
                    list(self.agent_pos[str(a)][0]) == list(self.agent_pos[str(a)][1])
                    for a in range(self.agent_size)
                )
                if all_have_initial_wait:
                    for a in range(self.agent_size):
                        self.agent_pos[str(a)] = self.agent_pos[str(a)][1:]

                output_data = {'schedule': {}}
                for agent in range(self.agent_size):
                    schedule = []
                    for step, pos in enumerate(self.agent_pos[str(agent)]):
                        schedule.append({'t': step, 'x': int(pos[0]), 'y': int(pos[1])})
                    output_data['schedule'][f'agent{agent}'] = schedule

                with open(self.output_file, 'w') as f:
                    yaml.dump(output_data, f)

                actual_steps = []
                for agent in range(self.agent_size):
                    path_length = len(self.agent_pos[str(agent)]) - 1
                    actual_steps.append(path_length)

                makespan = max(actual_steps)
                return makespan, elapsed_time, False

            # Timeout check
            if t > MAX_PLANNING_STEPS:
                elapsed_time = time.time() - start_time

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
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    planning = PlanningWithLivelock(
        agv_starts=instance['agv_starts'],
        tasks=instance['tasks'],
        output_file='yaml/output_test_livelock.yaml'
    )

    step, elapsed_time, dist = planning.main()
    print(f"Completion steps: {step}")
    print(f"Computation time: {elapsed_time:.4f} seconds")