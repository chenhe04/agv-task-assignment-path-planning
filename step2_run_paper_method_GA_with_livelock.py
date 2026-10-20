# step2_run_paper_method_GA_with_livelock.py
"""
目的：使用GA（遗传算法）进行任务分配
改进：集成活锁检测与解除机制（基于 step2_run_paper_method_GA.py）

对比原版：
- 原版：无活锁检测，A*+Q避障可能陷入循环震荡导致超时
- 改进版：自动检测并解除循环震荡活锁，提供详细的活锁统计信息
"""

import yaml
import time
import random as rand
import os
import random
import numpy as np
import planning_with_livelock as pnl
from taskassignment_GA import load_map_agv_task, genetic_algorithm

# ========== 可修改的GA参数 ==========
GA_GENERATIONS = 50
GA_POP_SIZE = 50
GA_MUTATION_RATE = 0.5
GA_RANDOM_SEED = 42
# ==================================

# ========== 实验配置 ==========
NUM_RUNS = 10
# ==================================

# ========== 活锁检测配置（新增） ==========
MAX_LIVELOCKS = 20
# ==================================

# ========== 地图文件自动检测 ==========
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


def run_paper_method_ga_with_livelock(
        agv_starts, tasks,
        generations=GA_GENERATIONS,
        pop_size=GA_POP_SIZE,
        mutation_rate=GA_MUTATION_RATE,
        random_seed=GA_RANDOM_SEED
):
    """
    使用GA进行任务分配优化，集成活锁检测

    返回值:
        step: 完成步数（不可行时返回 inf）
        valid_order: 最优任务顺序
        best_assignment: GA找到的分配方案
        total_time: 计算时间（GA搜索 + Planning执行时间）
        livelock_stats: 活锁统计信息
    """
    if random_seed is not None:
        rand.seed(random_seed)

    print(f"\n  [GA] 参数设置:")
    print(f"      世代数: {generations}")
    print(f"      个体数: {pop_size}")
    print(f"      变异率: {mutation_rate}")
    print(f"      随机种子: {random_seed}")
    print(f"      最大活锁次数: {MAX_LIVELOCKS}")

    print("\n  [GA] 搜索任务分配中...")
    ga_start = time.time()

    G, agvs, _, _, grid = load_map_agv_task(MAP_FILE)

    # ========== 坐标约定修复 ==========
    agv_states = [
        {'agv_id': i, 'position': (int(agv_starts[i][1]), int(agv_starts[i][0]))}
        for i in range(len(agv_starts))
    ]
    ga_tasks = [
        {'pickup': (int(t['pickup'][1]), int(t['pickup'][0])),
         'dropoff': (int(t['dropoff'][1]), int(t['dropoff'][0]))}
        for t in tasks
    ]

    best_assignment, best_cost = genetic_algorithm(
        G, ga_tasks, agv_states,
        generations=generations,
        pop_size=pop_size,
        mutation_rate=mutation_rate
    )
    ga_time = time.time() - ga_start

    print(f"  [GA] 最优成本（估计）: {best_cost:.1f} (计算时间: {ga_time:.2f}秒)")
    print(f"  [GA] 分配结果:")
    for agv_id, task_ids in enumerate(best_assignment):
        if task_ids:
            task_id = task_ids[0]
            task = tasks[task_id]
            print(f"      AGV{agv_id} → 任务{task_id}: {task['pickup']} → {task['dropoff']}")
        else:
            print(f"      AGV{agv_id} → 未分配")

    order = []
    for task_list in best_assignment:
        if task_list:
            order.append(task_list[0])
        else:
            order.append(-1)

    valid_order = [t for t in order if t >= 0]
    print(f"  [GA] 任务顺序: {valid_order}")

    print("\n  [Planning] 路径规划执行中（集成活锁检测）...")
    planning_start = time.time()

    ordered_tasks = [tasks[i] for i in valid_order]
    planning = pnl.PlanningWithLivelock(
        agv_starts=agv_starts,
        tasks=ordered_tasks,
        output_file='yaml/output_ga_with_livelock.yaml'
    )
    planning.livelock_detector.max_livelocks = MAX_LIVELOCKS

    step, elapsed_time, dist = planning.main()
    planning_time = time.time() - planning_start

    livelock_stats = planning.livelock_detector.get_statistics()

    if step > 0:
        print(f"  [Planning] 完成步数: {step} (计算时间: {planning_time:.2f}秒)")
        total_time = ga_time + planning_time
        print(f"  [合计] 计算时间: {total_time:.2f}秒")
        return step, valid_order, best_assignment, total_time, livelock_stats
    else:
        print(f"  [Planning] 警告: 未找到可行路径 (step={step})")
        return float('inf'), valid_order, best_assignment, ga_time + planning_time, livelock_stats


def run_paper_method_ga_simple(agv_starts, tasks):
    """简化接口"""
    return run_paper_method_ga_with_livelock(agv_starts, tasks)


if __name__ == "__main__":
    with open('instance.yaml', 'r') as f:
        instance = yaml.load(f, Loader=yaml.Loader)

    agv_starts = instance['agv_starts']
    tasks = instance['tasks']

    print("=" * 60)
    print("步骤2：GA任务分配（集成活锁检测版）")
    print("=" * 60)
    print(f"AGV起点: {agv_starts}")
    print(f"任务数: {len(tasks)}")
    print(f"AGV数: {len(agv_starts)}")
    print(f"使用地图: {MAP_FILE}")
    print(f"\n当前GA参数: 世代数={GA_GENERATIONS}, 个体数={GA_POP_SIZE}, 变异率={GA_MUTATION_RATE}")
    print(f"实验配置: 运行 {NUM_RUNS} 次，取最优结果")
    print(f"\n【改进亮点】")
    print(f"  ✓ 活锁检测：自动检测循环震荡活锁")
    print(f"  ✓ 活锁解除：打破对称性策略")
    print(f"  ✓ 详细日志：记录每次活锁的类型、时间、涉及AGV")
    print(f"  ✓ 统计信息：活锁次数、成功率、涉及AGV分布")

    # 多次运行取最优
    best_steps_all = float('inf')
    best_order_all = None
    best_assignment_all = None
    best_time_all = None
    best_livelock_stats = None
    all_steps = []
    all_times = []
    all_livelock_counts = []

    for run in range(NUM_RUNS):
        print(f"\n{'=' * 60}")
        print(f"第 {run + 1}/{NUM_RUNS} 次运行")
        print(f"{'=' * 60}")

        run_seed = GA_RANDOM_SEED + run

        best_steps, best_order, assignment, total_time, livelock_stats = \
            run_paper_method_ga_with_livelock(
                agv_starts, tasks,
                generations=GA_GENERATIONS,
                pop_size=GA_POP_SIZE,
                mutation_rate=GA_MUTATION_RATE,
                random_seed=run_seed
            )

        all_steps.append(best_steps)
        all_times.append(total_time)
        all_livelock_counts.append(livelock_stats['total_livelocks'])

        if best_steps < best_steps_all:
            best_steps_all = best_steps
            best_order_all = best_order
            best_assignment_all = assignment
            best_time_all = total_time
            best_livelock_stats = livelock_stats
            print(f"  ✓ 新的最优解: {best_steps} 步")

    # ========== 用最优顺序重新运行多次，确保输出最优路径 ==========
    if best_steps_all != float('inf') and best_order_all is not None:
        print(f"\n{'=' * 60}")
        print(f"  [保存最优路径] 用最优顺序 {best_order_all} 重新运行...")
        print(f"  [说明] 由于Q-learning的随机性，将运行10次取最优")
        print(f"{'=' * 60}")

        ordered_tasks = [instance['tasks'][i] for i in best_order_all]

        best_save_step = float('inf')
        best_save_time = None

        for trial in range(10):
            planning_start_trial = time.time()
            planning = pnl.PlanningWithLivelock(
                agv_starts=instance['agv_starts'],
                tasks=ordered_tasks,
                output_file='yaml/output_ga_temp.yaml'
            )
            step, elapsed_time, dist = planning.main()
            trial_time = time.time() - planning_start_trial

            if step > 0 and step < best_save_step:
                best_save_step = step
                best_save_time = trial_time
                import shutil
                shutil.copy('yaml/output_ga_temp.yaml', 'yaml/output_ga_with_livelock.yaml')
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒) ← 新的最优（已保存）")
            elif step == 0:
                print(f"    试验 {trial + 1}/10: 0 步 (不可行)")
            else:
                print(f"    试验 {trial + 1}/10: {step} 步 (时间: {trial_time:.2f}秒)")

        if os.path.exists('yaml/output_ga_temp.yaml'):
            os.remove('yaml/output_ga_temp.yaml')

        if best_save_step == float('inf'):
            print(f"\n  ⚠ 警告: 所有试验均未找到可行解，无法保存最优路径")
        else:
            print(f"\n  [保存完成] 最终步数: {best_save_step} 步 ✓")

            if best_save_step < best_steps_all:
                best_steps_all = best_save_step
                best_time_all = best_save_time
                print(f"  [更新] 全局最优步数更新为: {best_steps_all} 步")
    else:
        print(f"\n  ⚠ [GA法] 未找到任何可行解！")

    # 统计结果
    valid_steps = [s for s in all_steps if s != float('inf')]

    print(f"\n" + "=" * 60)
    print(f"最终结果（{NUM_RUNS}次运行统计）")
    print("=" * 60)

    if valid_steps:
        print(f"  最优顺序: {best_order_all}")
        print(f"  最少步数: {best_steps_all}")
        print(f"  平均步数: {np.mean(valid_steps):.2f}")
        print(f"  步数标准差: {np.std(valid_steps):.2f}")
        print(f"  步数范围: [{min(valid_steps)}, {max(valid_steps)}]")
        print(f"  成功率: {len(valid_steps)}/{NUM_RUNS} ({len(valid_steps) / NUM_RUNS * 100:.0f}%)")
        print(f"\n  最优运行时间: {best_time_all:.4f}秒")
        print(f"  平均运行时间: {np.mean(all_times):.4f}秒")

        print(f"\n  【活锁检测统计】")
        print(f"    平均活锁次数: {np.mean(all_livelock_counts):.2f}")
        print(f"    活锁次数范围: [{min(all_livelock_counts)}, {max(all_livelock_counts)}]")

        if best_livelock_stats:
            print(f"    最优运行活锁次数: {best_livelock_stats['total_livelocks']}")
            print(f"      - 循环震荡: {best_livelock_stats['oscillation_livelocks']}")
            print(f"      - 成功解除: {best_livelock_stats['successful_resolutions']}")
            if best_livelock_stats['involved_agents']:
                print(f"      - 涉及AGV: {best_livelock_stats['involved_agents']}")

        if best_assignment_all:
            print(f"  最优分配详情:")
            for agv_id, task_ids in enumerate(best_assignment_all):
                if task_ids:
                    task_id = task_ids[0]
                    task = tasks[task_id]
                    print(f"    AGV{agv_id} → 任务{task_id}: {task['pickup']} → {task['dropoff']}")
    else:
        print(f"\n  失败: 所有运行均未找到可行解")
        print(f"  平均运行时间: {np.mean(all_times):.4f} 秒")