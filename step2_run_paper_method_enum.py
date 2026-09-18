# step2_run_paper_method_enum.py
"""
目的：使用枚举法进行任务分配
"""

import yaml
import time
import os
import random
import numpy as np
from itertools import permutations
import planning as pn

# ========== 实验配置 ==========
NUM_RUNS = 1
# 默认运行次数
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


def run_paper_method(agv_starts, tasks):
    """
    运行三久代码，返回最小完成步数

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

    # 尝试所有任务排列
    for order in permutations(range(len(tasks))):
        # 按顺序排列任务
        ordered_tasks = [tasks[i] for i in order]

        # 直接调用 Planning 类
        planning = pn.Planning(
            agv_starts=agv_starts,
            tasks=ordered_tasks,
            output_file='yaml/output_enum.yaml'
        )

        step, elapsed_time, dist = planning.main()

        if step > 0 and step < best_steps:
            best_steps = step
            best_order = order
            print(f"    顺序 {order}: {step} 步 ✓")
        elif step == 0:
            print(f"    顺序 {order}: 无可行解")
        else:
            print(f"    顺序 {order}: {step} 步")

    # 计算时间结束
    total_time = time.time() - start_time

    return best_steps, best_order, total_time


if __name__ == "__main__":
    # 加载实例
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    print("=" * 60)
    print("步骤2：三久代码执行（枚举法）")
    print("=" * 60)
    print(f"AGV起点: {instance['agv_starts']}")
    print(f"任务: {instance['tasks']}")
    print(f"使用地图: {MAP_FILE}")
    print(f"\n实验配置: 运行 {NUM_RUNS} 次，取最优结果")
    print("\n尝试所有任务排列...")

    # 多次运行取最优
    best_steps_all = float('inf')
    best_order_all = None
    best_time_all = None
    all_steps = []
    all_times = []
    feasible_count = 0

    for run in range(NUM_RUNS):
        print(f"\n{'=' * 60}")
        print(f"第 {run + 1}/{NUM_RUNS} 次运行")
        print(f"{'=' * 60}")

        # 设置随机种子
        seed = BASE_SEED + run
        random.seed(seed)
        np.random.seed(seed)

        steps, order, elapsed = run_paper_method(
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
                output_file='yaml/output_enum_temp.yaml'
            )
            step, elapsed_time, dist = planning.main()
            trial_time = time.time() - planning_start_trial

            if step > 0 and step < best_save_step:
                best_save_step = step
                best_save_time = trial_time
                import shutil

                shutil.copy('yaml/output_enum_temp.yaml', 'yaml/output_enum.yaml')
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒) ← 新的最优（已保存）")
            elif step == 0:
                print(f"    试验 {trial + 1}/10: 0 步 (不可行)")
            else:
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒)")

        if os.path.exists('yaml/output_enum_temp.yaml'):
            os.remove('yaml/output_enum_temp.yaml')

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
    print(f"  失败次数: {NUM_RUNS - feasible_count}/{NUM_RUNS}")

    if feasible_count == 0:
        print(f"\n  ⚠ 警告: 枚举法在此实例上未找到任何可行解")
    elif feasible_count < NUM_RUNS:
        print(f"\n  ⚠ 注意: 枚举法稳定性不足 ({feasible_count / NUM_RUNS * 100:.1f}%)")
    else:
        print(f"\n  ✓ 枚举法在此实例上表现稳定 (100%可行)")

    if valid_steps:
        print(f"  最优顺序: {best_order_all}")
        print(f"  最少步数: {best_steps_all}")
        print(f"  平均步数: {np.mean(valid_steps):.2f}")
        print(f"  步数标准差: {np.std(valid_steps):.2f}")
        print(f"  步数范围: [{min(valid_steps)}, {max(valid_steps)}]")
        print(f"\n  最优运行时间: {best_time_all:.4f} 秒")
        print(f"  平均运行时间: {np.mean(all_times):.4f} 秒")
    else:
        print(f"  失败: 所有运行均未找到可行解")
        print(f"  平均运行时间: {np.mean(all_times):.4f} 秒")


