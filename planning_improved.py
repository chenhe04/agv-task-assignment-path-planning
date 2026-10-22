# planning_improved.py
"""
改进的路径规划模块：解决阻塞（Blocking）问题

核心改进：
- 在 A* 路径规划阶段，将停靠中的 AGV 视为动态障碍物
- A* 会规划出绕开停靠 AGV 的路径
- 从根本上避免"避碰后又回到原路径"的阻塞问题
"""

import numpy as np
import yaml
import time
import os
from astar_class import AStar

# ============================================================================
# 常量定义
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

# 动作空间：上、左、下、右、停留
ACTION_SET = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]])

# 观测节点（5×5网格的相对位置）
OBSERVATION_NODES = np.array([
    [0, 2], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1],
    [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [4, 2]
])

# 障碍物节点
OBSTACLE_NODES = np.array([[1, 2], [2, 1], [2, 3], [3, 2]])

# 途经点停留时间（步）
BASE_STOP_TIME = 1

# 装卸货时间（步）
LOAD_UNLOAD_TIME = 1

# 最大规划步数
MAX_PLANNING_STEPS = 300


class PlanningImproved:
    """
    改进的多智能体路径规划器

    核心改进：
    - 在 A* 阶段将停靠的 AGV 视为动态障碍物
    - 避免阻塞（Blocking）问题
    """

    def __init__(self, agv_starts=None, tasks=None, output_file='yaml/ga_improved_output.yaml'):
        """
        初始化路径规划器

        Args:
            agv_starts: AGV起始位置列表 [[x,y], [x,y], ...]
            tasks: 任务列表 [{'pickup':[x,y], 'dropoff':[x,y]}, ...]
            output_file: 输出文件路径
        """
        self.output_file = output_file

        # 加载地图
        with open(_MAP_FILE, 'r', encoding='utf-8') as f:
            map_data = yaml.load(f, Loader=yaml.Loader)

        self.col = map_data['map']['width']
        self.row = map_data['map']['height']
        self.agent_size = len(agv_starts)

        # 提取障碍物位置
        self.obs = []
        for y in range(self.row):
            for x in range(self.col):
                if map_data['map']['data'][y][x] == 1:
                    self.obs.append([x, y])

        # 初始化任务跟踪数组
        self.task_pos = np.zeros(self.agent_size, dtype=int)
        self.pick_or_place = np.full(self.agent_size, 0, dtype=int)

        # 初始化路径存储
        self.task_list = {}
        self.agent_pos = {}

        for agent in range(self.agent_size):
            initial_pos = agv_starts[agent]
            self.task_list[str(agent)] = [initial_pos]
            self.agent_pos[str(agent)] = [initial_pos]

        # 存储任务定义
        self.custom_tasks = tasks

        # 预分配任务给 AGV
        for agent_idx in range(min(len(tasks), self.agent_size)):
            task = tasks[agent_idx]
            self.task_list[str(agent_idx)].append(task['pickup'])
            self.task_list[str(agent_idx)].append(task['dropoff'])

        # 初始化状态网格（带边界填充）
        self.state = np.full((self.col, self.row), -2, dtype=int)
        for obstacle in self.obs:
            self.state[obstacle[0], obstacle[1]] = -1

        # 添加2层边界
        for _ in range(2):
            self.state = np.insert(self.state, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = np.insert(self.state.T, 0, -1, axis=0)
            self.state = np.insert(self.state, len(self.state), -1, axis=0)
            self.state = self.state.T

        # 加载预训练的 Q 函数
        self.q_function = np.load('yaml/qfunction.npy')

    def astar(self, t, agent, goal, start):
        """
        改进的A*路径规划：将停靠中的AGV视为动态障碍物

        核心改进：
        - 收集所有已完成任务、停在目标点的AGV
        - 将它们的位置作为动态障碍物传给A*
        - A*会规划出绕开停靠AGV的路径

        Args:
            t: 当前时间步
            agent: 当前AGV ID
            goal: 目标位置 [x, y]
            start: 起始位置 [x, y]

        Returns:
            bool: 是否找到路径
        """
        # ========== 收集停靠中的AGV位置（动态障碍物） ==========
        parked_agvs = []
        for i in range(self.agent_size):
            if i != agent:
                # 判断AGV i 是否已完成任务
                if (self.task_pos[i] == len(self.task_list[str(i)]) - 1 and
                        t + BASE_STOP_TIME < len(self.agent_pos[str(i)])):
                    final_goal = np.array(self.task_list[str(i)][-1])
                    current_pos_i = np.array(self.agent_pos[str(i)][t + BASE_STOP_TIME])

                    # 如果停在最终目标点 → 视为停靠中
                    if np.all(current_pos_i == final_goal):
                        parked_agvs.append([int(current_pos_i[0]), int(current_pos_i[1])])
        # ============================================================

        # ========== 合并静态障碍物和动态障碍物 ==========
        all_obstacles = self.obs.copy()
        all_obstacles.extend(parked_agvs)
        # ============================================================

        # 调用A*，传入所有障碍物（静态 + 动态）
        a_star = AStar(self.col, self.row, goal, start, False, all_obstacles)
        path = a_star.main()

        if path:
            for node in path:
                self.agent_pos[str(agent)].append([node.x, node.y])

            self.agent_pos[str(agent)].append([path[-1].x, path[-1].y])

            return True
        return False

    def get_state(self, t, agent):
        """
        获取局部观测状态（恢复原版逻辑）

        因为停靠的AGV已经在A*阶段处理，
        这里不需要特殊处理，所有AGV都参与冲突检测。
        """
        passing = np.zeros(len(ACTION_SET) - 1, dtype=int)
        state_ = np.copy(self.state)

        # 计算当前移动方向
        a = np.array(self.agent_pos[str(agent)][t + 1]) - \
            np.array(self.agent_pos[str(agent)][t])
        idx = 0
        for i in range(len(ACTION_SET)):
            if np.all(ACTION_SET[i] - a == 0):
                idx = i
                break

        # 检测其他AGV和障碍物
        for i in range(self.agent_size):
            if i != agent:
                # 所有AGV都参与冲突检测（不再特殊处理已完成的AGV）
                for j in range(len(ACTION_SET) - 1):
                    temp = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[j]
                    if (np.all(np.array(self.agent_pos[str(i)][t]) - temp == 0) and
                            np.all(np.array(self.agent_pos[str(i)][t + 1]) -
                                   np.array(self.agent_pos[str(agent)][t]) == 0)):
                        passing[j] = 1
                state_[self.agent_pos[str(i)][t + 1][0] + 2,
                       self.agent_pos[str(i)][t + 1][1] + 2] = 0

        # 提取5×5局部观测
        local_state = state_[
            self.agent_pos[str(agent)][t][0]:self.agent_pos[str(agent)][t][0] + 5,
            self.agent_pos[str(agent)][t][1]:self.agent_pos[str(agent)][t][1] + 5
        ]

        # 旋转到当前移动方向
        for _ in range(idx):
            local_state = np.rot90(local_state, -1)

        return local_state, idx, passing

    def main(self):
        """
        执行主规划循环

        Returns:
            tuple: (makespan, elapsed_time, distance)
        """
        t = 0
        start_time = time.time()

        while True:
            # ========================================================================
            # Phase 1: 任务分配和A*路径规划
            # ========================================================================
            for agent in range(self.agent_size):
                # 确保路径缓冲区长度足够
                while len(self.agent_pos[str(agent)]) <= t + BASE_STOP_TIME:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])
                target_pos = np.array(self.task_list[str(agent)][self.task_pos[agent]])

                # 检查是否到达当前目标
                if np.all(current_pos - target_pos == 0):
                    # 如果有下一个目标，继续前进
                    if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                        next_target = self.task_list[str(agent)][self.task_pos[agent] + 1]
                        start_pos = self.agent_pos[str(agent)][t + BASE_STOP_TIME]

                        # 规划到下一个目标的路径
                        success = self.astar(t, agent, next_target, start_pos)

                        # 只有规划成功才推进任务位置
                        if success:
                            self.task_pos[agent] += 1
                    else:
                        # 所有任务完成，停在最终位置
                        self.agent_pos[str(agent)].append(
                            self.agent_pos[str(agent)][t + BASE_STOP_TIME]
                        )

            # ========================================================================
            # Phase 2: 基于Q-learning的避碰
            # ========================================================================
            for agent in range(self.agent_size):
                # 确保路径缓冲区长度足够
                while len(self.agent_pos[str(agent)]) <= t + 1:
                    self.agent_pos[str(agent)].append(self.agent_pos[str(agent)][-1])

                # 获取局部观测状态
                local_state, rotate, passing = self.get_state(t, agent)

                # 计算目标方向
                goal = np.array(self.task_list[str(agent)][self.task_pos[agent]]) - \
                       np.array(self.agent_pos[str(agent)][t])

                # 确定偏好方向
                if abs(goal[0]) >= abs(goal[1]):
                    muki = 0 if goal[0] < 0 else 2
                else:
                    muki = 1 if goal[1] < 0 else 3
                muki = (muki - rotate) % 4

                # 编码状态
                agv_node = '0000000000000'
                obs_node = '0000'
                passing_ = np.zeros(len(ACTION_SET) - 1, dtype=int)

                # 编码AGV位置
                for i in range(len(OBSERVATION_NODES)):
                    if local_state[OBSERVATION_NODES[i][0]][OBSERVATION_NODES[i][1]] == 0:
                        agv_node = agv_node[:i] + '1' + agv_node[i + 1:]

                # 编码障碍物位置
                for i in range(len(OBSTACLE_NODES)):
                    if local_state[OBSTACLE_NODES[i][0]][OBSTACLE_NODES[i][1]] == -1:
                        obs_node = obs_node[:i] + '1' + obs_node[i + 1:]

                # 编码目标方向和交叉冲突信息
                goal_node = format(muki, '02b')
                for p in range(len(passing)):
                    if passing[p] == 1:
                        passing_[(4 + p - rotate) % len(passing)] = 1

                passing__ = passing_[0] * 8 + passing_[1] * 4 + \
                            passing_[2] * 2 + passing_[3]
                passing__ = format(passing__, '04b')
                state_number = int(agv_node + obs_node + goal_node + passing__, 2)

                # 查询Q表，选择最优动作
                q = self.q_function[state_number]
                max_value = np.max(q)
                max_index = [a for a in range(len(q)) if q[a] == max_value]
                action = max_index[np.random.randint(0, len(max_index))]

                # 应用避碰操作（如果需要）
                if (action != 0 and
                        not np.all(np.array(self.agent_pos[str(agent)][t]) -
                                   np.array(self.agent_pos[str(agent)][t + 1]) == 0)):
                    if action != 4:
                        action = (action + rotate) % 4

                    # 截断路径，应用新动作
                    self.agent_pos[str(agent)] = self.agent_pos[str(agent)][:t + 1]
                    new_pos = np.array(self.agent_pos[str(agent)][t]) + ACTION_SET[action]
                    self.agent_pos[str(agent)].append(new_pos.tolist())

                    # 从新位置重新规划
                    self.astar(t, agent,
                               self.task_list[str(agent)][self.task_pos[agent]],
                               new_pos.tolist())

            t += 1

            # ========================================================================
            # Phase 3: 终止条件检查
            # ========================================================================
            all_completed = True
            for agent in range(self.agent_size):
                # 检查是否有未完成的任务
                if self.task_pos[agent] < len(self.task_list[str(agent)]) - 1:
                    all_completed = False
                    break

                # 验证是否到达最终目标
                current_pos = np.array(self.agent_pos[str(agent)][t + BASE_STOP_TIME])
                final_goal = np.array(self.task_list[str(agent)][-1])
                if not np.all(current_pos == final_goal):
                    all_completed = False
                    break

            if all_completed:
                elapsed_time = time.time() - start_time

                # ====================================================================
                # Phase 4: 解验证（后验证）
                # ====================================================================
                all_tasks_valid = True

                for task_idx, task in enumerate(self.custom_tasks):
                    pickup = tuple(task['pickup'])
                    dropoff = tuple(task['dropoff'])

                    # 识别哪个AGV被分配了这个任务
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
                        print(f"[WARN] 任务{task_idx}未分配给任何AGV: "
                              f"pickup={pickup}, dropoff={dropoff}")
                        break

                    # 验证分配给该AGV的任务执行情况
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
                        print(f"[WARN] 任务{task_idx} (AGV{task_assigned_to_agent}) "
                              f"未到达pickup: {pickup}")
                        break

                    if dropoff_time == -1:
                        all_tasks_valid = False
                        print(f"[WARN] 任务{task_idx} (AGV{task_assigned_to_agent}) "
                              f"未到达dropoff: {dropoff}")
                        break

                    # 验证pickup点的停留时间约束
                    pickup_stay_count = 0
                    for step_idx in range(pickup_time, len(path)):
                        if tuple(path[step_idx]) == pickup:
                            pickup_stay_count += 1
                        else:
                            break

                    required_stay = BASE_STOP_TIME + LOAD_UNLOAD_TIME
                    if pickup_stay_count < required_stay:
                        all_tasks_valid = False
                        print(f"[WARN] 任务{task_idx} (AGV{task_assigned_to_agent}) "
                              f"pickup停留时间不足: {pickup_stay_count} 步 "
                              f"(要求: {required_stay} 步)")
                        break

                    # 验证dropoff点的停留时间约束
                    dropoff_stay_count = 0
                    for step_idx in range(dropoff_time, len(path)):
                        if tuple(path[step_idx]) == dropoff:
                            dropoff_stay_count += 1
                        else:
                            break

                    if dropoff_stay_count < required_stay:
                        all_tasks_valid = False
                        print(f"[WARN] 任务{task_idx} (AGV{task_assigned_to_agent}) "
                              f"dropoff停留时间不足: {dropoff_stay_count} 步 "
                              f"(要求: {required_stay} 步)")
                        break

                # 拒绝不可行解
                if not all_tasks_valid:
                    print("[ERROR] 检测到任务未完成，返回0步（不可行解）")
                    return 0, elapsed_time, False

                # 归一化：消除起点的强制1步启动等待
                all_have_initial_wait = all(
                    len(self.agent_pos[str(a)]) >= 2 and
                    list(self.agent_pos[str(a)][0]) == list(self.agent_pos[str(a)][1])
                    for a in range(self.agent_size)
                )
                if all_have_initial_wait:
                    for a in range(self.agent_size):
                        self.agent_pos[str(a)] = self.agent_pos[str(a)][1:]

                # 保存有效解到文件
                output_data = {'schedule': {}}
                for agent in range(self.agent_size):
                    schedule = []
                    for step, pos in enumerate(self.agent_pos[str(agent)]):
                        schedule.append({'t': step, 'x': int(pos[0]), 'y': int(pos[1])})
                    output_data['schedule'][f'agent{agent}'] = schedule

                with open(self.output_file, 'w') as f:
                    yaml.dump(output_data, f)

                # 计算makespan（所有AGV的最大完成时间）
                actual_steps = []
                for agent in range(self.agent_size):
                    path_length = len(self.agent_pos[str(agent)]) - 1
                    actual_steps.append(path_length)

                makespan = max(actual_steps)
                return makespan, elapsed_time, False

            # 超时检查
            if t > MAX_PLANNING_STEPS:
                elapsed_time = time.time() - start_time

                # 保存部分解用于调试
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
    # 使用instance配置进行测试
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    planning = PlanningImproved(
        agv_starts=instance['agv_starts'],
        tasks=instance['tasks'],
        output_file='yaml/output_test_improved.yaml'
    )

    step, elapsed_time, dist = planning.main()
    print(f"完成步数: {step}")
    print(f"计算时间: {elapsed_time:.4f} 秒")