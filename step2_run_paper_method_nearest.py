# step2_run_paper_method_nearest.py
"""
目的：使用最近邻法进行任务分配
"""

import yaml
import time
import os
import random
import numpy as np
import planning as pn
from astar_class import AStar

# ========== 实验配置 ==========
NUM_RUNS = 10  # 默认运行次数
BASE_SEED = 42  # 基础随机种子
# ==================================

# ========== 地图文件自动检测 ==========
# 优先级顺序：
# 1. custom_map.yaml (通用)
# 2. map_small_traditional_aisle.yaml (10x10)
# 3. map_medium_traditional_aisle.yaml (15x15)
# 4. map_large_traditional_aisle.yaml (20x20)
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
        "未找到地图文件！请先运行 step1_create_instance.py 生成地图。\n"
        "期望的文件: custom_map.yaml 或 map_{small/medium/large}_traditional_aisle.yaml"
    )
else:
    print(f"[自动检测] 使用地图文件: {MAP_FILE}")


# ==================================


def get_astar_distance(start, goal, obstacles, width, height):
    """使用A*计算两点间距离"""
    if start == goal:
        return 0
    try:
        astar = AStar(
            cols=width, rows=height,
            start=[start[0], start[1]],
            end=[goal[0], goal[1]],
            obstacle_ratio=False,
            obstacle_list=obstacles
        )
        path = astar.main()
        return len(path) - 1 if path else float('inf')
    except:
        return float('inf')


def run_paper_method_nearest(agv_starts, tasks):
    """
    运行最近邻法，返回最小完成步数

    参数:
        agv_starts: list of [x,y], AGV起点列表
        tasks: list of {'pickup':[x,y], 'dropoff':[x,y]}, 任务列表

    返回值:
        best_steps: 最小完成步数（无解时返回float('inf')）
        best_order: 最优任务顺序（无解时返回None）
        total_time: 计算时间（秒）
    """

    best_steps = float('inf')
    best_order = None

    # 计算时间开始
    start_time = time.time()

    # ========== 新增：加载地图获取障碍物信息 ==========
    with open(MAP_FILE, 'r', encoding='utf-8') as f:
        map_data = yaml.load(f, Loader=yaml.Loader)

    width = map_data['map']['width']
    height = map_data['map']['height']
    obstacles = []
    for y in range(height):
        for x in range(width):
            if map_data['map']['data'][y][x] == 1:
                obstacles.append([x, y])
    # ==========================================

    # 计算距离矩阵
    n_agvs = len(agv_starts)
    n_tasks = len(tasks)

    # ========== 修正：构建 AGV到任务的总距离矩阵（A*实际距离）==========
    print(f"  [NN] 正在计算A*距离矩阵...")
    dist_matrix = np.zeros((n_agvs, n_tasks))
    for i in range(n_agvs):
        for j in range(n_tasks):
            pickup = tasks[j]['pickup']
            dropoff = tasks[j]['dropoff']

            # 计算总距离：起点→取货点 + 取货点→送货点
            dist_to_pickup = get_astar_distance(
                agv_starts[i], pickup, obstacles, width, height
            )
            dist_pickup_to_dropoff = get_astar_distance(
                pickup, dropoff, obstacles, width, height
            )
            dist_matrix[i, j] = dist_to_pickup + dist_pickup_to_dropoff
    # ==========================================

    # 贪心分配：每次选择距离最近的AGV-任务对
    assigned_tasks = set()
    assignment = [None] * n_agvs  # assignment[agv_idx] = task_idx，按AGV索引记录
    remaining_agvs = list(range(n_agvs))

    for _ in range(min(n_agvs, n_tasks)):
        min_dist = float('inf')
        best_agv_idx = None
        best_task_idx = None

        for agv_idx in remaining_agvs:
            for task_idx in range(n_tasks):
                if task_idx not in assigned_tasks:
                    if dist_matrix[agv_idx, task_idx] < min_dist:
                        min_dist = dist_matrix[agv_idx, task_idx]
                        best_agv_idx = agv_idx
                        best_task_idx = task_idx

        if best_task_idx is not None:
            assigned_tasks.add(best_task_idx)
            assignment[best_agv_idx] = best_task_idx  # 关键：记录“该AGV做该任务”
            remaining_agvs.remove(best_agv_idx)

    # 按AGV索引构建任务顺序：order[i] = AGV i 要执行的任务，与 Planning 的直接映射一致
    order = [assignment[i] for i in range(n_agvs) if assignment[i] is not None]

    print(f"  [最近邻法] 任务顺序: {order}")
    for idx, task_idx in enumerate(order):
        task = tasks[task_idx]
        print(f"      AGV{idx}  任务{task_idx}: {task['pickup']} → {task['dropoff']}")

    # ========== 路径规划可行性检查 ==========
    ordered_tasks = [tasks[i] for i in order]

    planning = pn.Planning(
        agv_starts=agv_starts,
        tasks=ordered_tasks,
        output_file='yaml/output_nn.yaml'
    )

    step, elapsed_time, dist = planning.main()

    # 检查是否找到可行解
    if step == 0:
        print(f"\n  ⚠ [可行性检查] NN方法在此实例上无可行解！")
        print(f"     原因: 任务分配导致路径冲突或死锁")
        print(f"     建议: 此启发式方法不适用于通道式仓库地图")
        total_time = time.time() - start_time
        return float('inf'), None, total_time

    # 如果找到可行解
    best_steps = step
    best_order = tuple(order)
    print(f"  [最近邻法] 完成步数: {best_steps} 步 ✓")

    # 计算时间结束
    total_time = time.time() - start_time

    return best_steps, best_order, total_time


def run_paper_method_nearest_with_tuning(agv_starts, tasks, n_trials=1):
    """
    最近邻法是确定性的，n_trials参数无意义
    为保持接口一致性而实现
    """
    return run_paper_method_nearest(agv_starts, tasks)


if __name__ == "__main__":
    # 加载实例
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    print("=" * 60)
    print("步骤2：三久代码执行（最近邻法）")
    print("=" * 60)
    print(f"AGV起点: {instance['agv_starts']}")
    print(f"任务数: {len(instance['tasks'])}")
    print(f"AGV数: {len(instance['agv_starts'])}")
    print(f"使用地图: {MAP_FILE}")
    print(f"\n实验配置: 运行 {NUM_RUNS} 次，取最优结果")

    # 多次运行取最优
    best_steps_all = float('inf')
    best_order_all = None
    best_time_all = None
    all_steps = []
    all_times = []
    feasible_count = 0  # 记录可行解次数

    for run in range(NUM_RUNS):
        print(f"\n{'=' * 60}")
        print(f"第 {run + 1}/{NUM_RUNS} 次运行")
        print(f"{'=' * 60}")

        # 设置随机种子
        seed = BASE_SEED + run
        random.seed(seed)
        np.random.seed(seed)

        steps, order, elapsed = run_paper_method_nearest(
            instance['agv_starts'],
            instance['tasks']
        )

        all_steps.append(steps)
        all_times.append(elapsed)

        if steps != float('inf'):
            feasible_count += 1
            if steps < best_steps_all:
                best_steps_all = steps
                best_order_all = order
                best_time_all = elapsed
                print(f"  ✓ 新的最优解: {steps} 步")

    # ========== 用最优顺序重新运行多次，确保输出最优路径 ==========
    if best_steps_all != float('inf') and best_order_all is not None:
        print(f"\n{'=' * 60}")
        print(f"  [保存最优路径] 用最优顺序 {best_order_all} 重新运行...")
        print(f"  [说明] 由于Q-learning的随机性，将运行10次取最优")
        print(f"{'=' * 60}")

        ordered_tasks = [instance['tasks'][i] for i in best_order_all]

        # 运行多次，取最优的那次（排除0步的不可行解）
        best_save_step = float('inf')
        best_save_time = None

        for trial in range(10):
            planning_start_trial = time.time()
            planning = pn.Planning(
                agv_starts=instance['agv_starts'],
                tasks=ordered_tasks,
                output_file='yaml/output_nn_temp.yaml'
            )
            step, elapsed_time, dist = planning.main()
            trial_time = time.time() - planning_start_trial

            if step > 0 and step < best_save_step:
                best_save_step = step
                best_save_time = trial_time
                import shutil

                shutil.copy('yaml/output_nn_temp.yaml', 'yaml/output_nn.yaml')
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒) ← 新的最优（已保存）")
            elif step == 0:
                print(f"    试验 {trial + 1}/10: 0 步 (不可行)")
            else:
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒)")

        if os.path.exists('yaml/output_nn_temp.yaml'):
            os.remove('yaml/output_nn_temp.yaml')

        # 检查是否找到可行解
        if best_save_step == float('inf'):
            print(f"\n  ⚠ 警告: 所有试验均未找到可行解，无法保存最优路径")
        else:
            print(f"\n  [保存完成] 最终步数: {best_save_step} 步 ✓")

            # 【关键修复】更新全局最优结果
            if best_save_step < best_steps_all:
                best_steps_all = best_save_step
                best_time_all = best_save_time
                print(f"  [更新] 全局最优步数更新为: {best_steps_all} 步")
    else:
        print(f"\n  ⚠ [枚举法] 未找到任何可行解！")
    # ==========================================
    # 统计结果
    valid_steps = [s for s in all_steps if s != float('inf')]

    print(f"\n" + "=" * 60)
    print(f"最终结果（{NUM_RUNS}次运行统计）")
    print("=" * 60)

    # ========== 可行性分析 ==========
    print(f"\n【可行性分析】")
    print(f"  可行解次数: {feasible_count}/{NUM_RUNS} ({feasible_count / NUM_RUNS * 100:.1f}%)")

    if feasible_count == 0:
        print(f"\n  ⚠ 结论: NN方法在此实例上完全不可行")
    elif feasible_count < NUM_RUNS:
        print(f"\n  ⚠ 警告: NN方法稳定性较差")
        print(f"     仅在 {feasible_count}/{NUM_RUNS} 次运行中找到可行解")
    else:
        print(f"\n  ✓ NN方法在此实例上表现稳定")
    # ==========================================

    if valid_steps:
        print(f"\n【性能指标】")
        print(f"  最优顺序: {best_order_all}")
        print(f"  最少步数: {best_steps_all}")
        print(f"  平均步数: {np.mean(valid_steps):.2f}")
        print(f"  步数标准差: {np.std(valid_steps):.2f}")
        print(f"  步数范围: [{min(valid_steps)}, {max(valid_steps)}]")
        print(f"\n  最优运行时间: {best_time_all:.4f} 秒")
        print(f"  平均运行时间: {np.mean(all_times):.4f} 秒")
    else:
        print(f"\n  失败: 所有运行均未找到可行解")
        print(f"  平均运行时间: {np.mean(all_times):.4f} 秒")

