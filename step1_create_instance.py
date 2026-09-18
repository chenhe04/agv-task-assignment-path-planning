# step1_create_instance.py
"""
自定义地图生成模块
采用传统通道式仓库布局（Traditional Aisle Layout）
- 平行货架行 + 垂直拣货通道 + 中央水平主通道
- 工业标准布局，80%+真实仓库使用

支持三种实验规模：
  - small:   10x10, 3 AGV, 3 任务（最优性验证）
  - medium:  15x15, 5 AGV, 5 任务（中规模测试）
  - large:   20x20, 7 AGV, 7 任务（大规模测试）
"""

import yaml
import random

# ========== 实验规模选择 ==========
# 'small'  - 10x10, 3 AGV, 3 任务（最优性验证，完整MIP可解）
# 'medium' - 15x15, 5 AGV, 5 任务（中规模测试，GA vs Nearest Neighbor）
# 'large'  - 20x20, 7 AGV, 7 任务（大规模测试，GA性能极限）
EXPERIMENT_SIZE = 'large'  # 可选: 'small', 'medium', 'large'
# ==================================


# 实验规模配置（包含布局参数）
SIZE_CONFIG = {
    'small': {
        'width': 10,
        'height': 10,
        'num_agvs': 3,
        'num_tasks': 3,
        'num_vertical_aisles': 3,      # 3条垂直通道
        'num_main_corridors': 1,       # 1条中央主干道
        'aisle_width': 1,              # 窄通道
        'corridor_width': 1            # 窄主干道
    },
    'medium': {
        'width': 15,
        'height': 15,
        'num_agvs': 5,
        'num_tasks': 5,
        'num_vertical_aisles': 4,      # 4条垂直通道
        'num_main_corridors': 2,       # 2条主干道
        'aisle_width': 1,              # 窄通道
        'corridor_width': 1
    },
    'large': {
        'width': 20,
        'height': 20,
        'num_agvs': 7,
        'num_tasks': 7,
        'num_vertical_aisles': 6,      # 6条垂直通道
        'num_main_corridors': 3,       # 3条主干道
        'aisle_width': 1,
        'corridor_width': 1
    },
}


def get_traditional_aisle_obstacles(width, height, num_vertical_aisles, num_main_corridors, aisle_width,
                                    corridor_width):
    """
    传统通道式仓库布局（工业标准）

    特点：
    - 平行货架行（Storage Rows）
    - 垂直拣货通道（Picking Aisles）
    - 水平主通道（Main Corridor）
    - 边界环形通道

    优势：
    ✓ 路径可预测，便于A*规划
    ✓ 空间利用率高（60-70%）
    ✓ 适合高密度存储
    ✓ 80%+真实仓库采用此布局

    参数:
        width: 地图宽度
        height: 地图高度
        num_vertical_aisles: 垂直通道数量
        num_main_corridors: 水平主干道数量
        aisle_width: 垂直通道宽度（格子数）
        corridor_width: 水平主干道宽度（格子数）

    返回:
        obstacles: 障碍物坐标列表 [[x,y], ...]
    """
    obstacles = []

    # 布局参数
    border_margin = 1 if width <= 10 else 2  # 边界保留距离

    # 计算水平主干道位置
    if num_main_corridors == 1:
        main_corridor_y_positions = [height // 2]
    else:
        # 均匀分布多条主干道
        spacing = height // (num_main_corridors + 1)
        main_corridor_y_positions = [spacing * (i + 1) for i in range(num_main_corridors)]

    # 计算垂直通道位置和间距
    usable_width = width - 2 * border_margin
    aisle_spacing = usable_width // num_vertical_aisles

    # 生成货架障碍物
    for x in range(border_margin, width - border_margin):
        for y in range(border_margin, height - border_margin):
            # 检查是否在水平主干道区域内
            in_horizontal_corridor = False
            for cy in main_corridor_y_positions:
                if abs(y - cy) <= corridor_width // 2:
                    in_horizontal_corridor = True
                    break

            if in_horizontal_corridor:
                continue

            # 检查是否在垂直通道内
            in_vertical_aisle = False
            for i in range(num_vertical_aisles):
                aisle_x = border_margin + i * aisle_spacing + aisle_spacing // 2
                if abs(x - aisle_x) <= aisle_width // 2:
                    in_vertical_aisle = True
                    break

            # 如果不在任何通道内，则放置货架
            if not in_vertical_aisle:
                obstacles.append([x, y])

    # 去重
    obstacles = list(set(tuple(o) for o in obstacles))
    obstacles = [list(o) for o in obstacles]

    return obstacles

def get_agv_starts(width, height, num_agvs, obstacles):
    """
    获取AGV起点（避开障碍物）
    优先选择角落和边缘位置，确保AGV之间有足够的空间
    """
    obstacle_set = set(tuple(o) for o in obstacles)

    # 候选起点列表（四个角的内侧 + 边的中点）
    candidates = [
        [1, 1], [1, height - 2], [width - 2, 1], [width - 2, height - 2],
        [width // 2, 1], [width // 2, height - 2],
        [1, height // 2], [width - 2, height // 2],
    ]

    # 对于大地图，增加更多候选点
    if width >= 15:
        candidates.extend([
            [3, 3], [3, height - 4], [width - 4, 3], [width - 4, height - 4],
            [width // 3, 1], [2 * width // 3, 1],
            [width // 3, height - 2], [2 * width // 3, height - 2],
        ])

    # 过滤障碍物
    valid_starts = []
    for p in candidates:
        if tuple(p) not in obstacle_set and p not in valid_starts:
            valid_starts.append(p)

    # 如果不够，扩大搜索范围
    if len(valid_starts) < num_agvs:
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if len(valid_starts) >= num_agvs:
                    break
                if tuple([x, y]) not in obstacle_set and [x, y] not in valid_starts:
                    valid_starts.append([x, y])

    return valid_starts[:num_agvs]


def is_crossroad(x, y, obstacles, width, height):
    """
    检查位置是否是十字路口

    参数:
        x, y: 待检查的坐标
        obstacles: 障碍物列表
        width, height: 地图尺寸

    返回:
        True: 是十字路口（4个方向都连通）
        False: 不是十字路口
    """
    obstacle_set = set(tuple(o) for o in obstacles)

    # 检查四个方向的连通性
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # 上下左右
    free_count = 0

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # 检查边界
        if 0 <= nx < width and 0 <= ny < height:
            # 检查是否是障碍物
            if (nx, ny) not in obstacle_set:
                free_count += 1

    # 4个方向都连通 = 十字路口
    return free_count == 4


def get_tasks(width, height, num_tasks, obstacles, agv_starts):
    """
    获取任务点（避开障碍物和AGV起点，且不重复）
    使用固定随机种子保证可复现性
    新增：避免将送货点设置在十字路口
    """
    obstacle_set = set(tuple(o) for o in obstacles)
    start_set = set(tuple(s) for s in agv_starts)

    # 所有已被占用的点（起点）
    used_points = set(start_set)

    # 候选点池
    all_candidates = []
    margin = 2

    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            if (x, y) not in obstacle_set and (x, y) not in used_points:
                all_candidates.append([x, y])

    # 打乱顺序（固定种子保证可复现）
    random.seed(42)
    random.shuffle(all_candidates)

    tasks = []
    for i in range(num_tasks):
        if len(all_candidates) >= 2:
            pickup = all_candidates.pop(0)

            # ========== 新增：避免十字路口作为送货点 ==========
            # 从候选池中选择一个不是十字路口的送货点
            dropoff = None
            for j in range(len(all_candidates)):
                candidate = all_candidates[j]
                if not is_crossroad(candidate[0], candidate[1], obstacles, width, height):
                    dropoff = all_candidates.pop(j)
                    break

            # 如果没找到非十字路口的点，就用第一个（兜底）
            if dropoff is None:
                dropoff = all_candidates.pop(0)
                print(f"  ⚠ 警告: 任务{i}的送货点{dropoff}位于十字路口")
            # ==========================================
        else:
            # 备用方案
            pickup = [margin, height // 2]
            dropoff = [width - margin - 1, height // 2]

        tasks.append({
            'pickup': pickup,
            'dropoff': dropoff
        })

        used_points.add(tuple(pickup))
        used_points.add(tuple(dropoff))

    return tasks

def print_grid(grid, width, height, agv_starts, tasks):
    """打印网格预览"""
    print("\n网格预览:")
    print("-" * (width * 2 + 4))
    for y in range(height):
        row = ""
        for x in range(width):
            if [x, y] in agv_starts:
                row += "S "
            elif [x, y] in [t['pickup'] for t in tasks]:
                row += "P "
            elif [x, y] in [t['dropoff'] for t in tasks]:
                row += "D "
            elif grid[y][x] == 1:
                row += "# "
            else:
                row += ". "
        print(row)
    print("-" * (width * 2 + 4))
    print("图例: S=AGV起点, P=取货点, D=送货点, #=障碍物, .=可通行")


def create_instance():
    """
    创建测试实例（传统通道式仓库布局）
    """
    # 获取配置
    config = SIZE_CONFIG[EXPERIMENT_SIZE]
    width = config['width']
    height = config['height']
    num_agvs = config['num_agvs']
    num_tasks = config['num_tasks']
    num_vertical_aisles = config['num_vertical_aisles']
    num_main_corridors = config['num_main_corridors']
    aisle_width = config['aisle_width']
    corridor_width = config['corridor_width']

    print("=" * 60)
    print("创建测试实例 - 传统通道式仓库布局")
    print("=" * 60)
    print(f"实验规模: {EXPERIMENT_SIZE}")
    print(f"布局类型: traditional_aisle（工业标准）")
    print(f"地图大小: {width} x {height}")
    print(f"AGV数量: {num_agvs}")
    print(f"任务数量: {num_tasks}")
    print(f"垂直通道: {num_vertical_aisles}条 (宽度={aisle_width})")
    print(f"水平主干道: {num_main_corridors}条 (宽度={corridor_width})")

    # 获取障碍物
    obstacles = get_traditional_aisle_obstacles(
        width, height,
        num_vertical_aisles, num_main_corridors,
        aisle_width, corridor_width
    )
    print(f"障碍物数量: {len(obstacles)}")
    print(f"空间利用率: {len(obstacles) / (width * height) * 100:.1f}%")

    # 获取AGV起点
    agv_starts = get_agv_starts(width, height, num_agvs, obstacles)
    print(f"\nAGV起点:")
    for i, start in enumerate(agv_starts):
        print(f"  AGV{i}: {start}")

    # 获取任务
    tasks = get_tasks(width, height, num_tasks, obstacles, agv_starts)
    print(f"\n任务:")
    for i, task in enumerate(tasks):
        print(f"  任务{i}: 取货{task['pickup']} -> 送货{task['dropoff']}")

    # 验证
    obstacle_set = set(tuple(o) for o in obstacles)
    all_valid = True

    # 检查AGV起点
    for i, start in enumerate(agv_starts):
        if tuple(start) in obstacle_set:
            print(f"  ✗ AGV{i} 起点 {start} 在障碍物上！")
            all_valid = False

    # 检查任务点
    task_points = []
    for task in tasks:
        task_points.append(tuple(task['pickup']))
        task_points.append(tuple(task['dropoff']))

    if len(task_points) != len(set(task_points)):
        print(f"  ✗ 任务点有重复！")
        all_valid = False

    for i, task in enumerate(tasks):
        if tuple(task['pickup']) in obstacle_set:
            print(f"  ✗ 任务{i} 取货点 {task['pickup']} 在障碍物上！")
            all_valid = False
        if tuple(task['dropoff']) in obstacle_set:
            print(f"  ✗ 任务{i} 送货点 {task['dropoff']} 在障碍物上！")
            all_valid = False

    if not all_valid:
        print("\n⚠ 警告: 存在问题，请调整布局或参数")
        return

    print("\n✓ 验证通过")

    # 构建网格
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for x, y in obstacles:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 1

    # 保存地图文件（planning.py使用）
    map_data = {
        'map': {
            'width': width,
            'height': height,
            'data': grid
        },
        'agvs': [{'position': start} for start in agv_starts],
        'tasks': {
            'initial': [{'pickup': t['pickup'], 'dropoff': t['dropoff']} for t in tasks]
        }
    }

    with open('custom_map.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(map_data, f, allow_unicode=True, default_flow_style=False)

    # 保存可视化专用地图文件（与 visualize_paths.py 兼容）
    visual_map_file = f'map_{EXPERIMENT_SIZE}_traditional_aisle.yaml'
    with open(visual_map_file, 'w', encoding='utf-8') as f:
        yaml.dump(map_data, f, allow_unicode=True, default_flow_style=False)

    # 保存实例配置
    instance = {
        'name': f'{EXPERIMENT_SIZE}_traditional_aisle_{width}x{height}_{num_agvs}agv_{num_tasks}task',
        'experiment_size': EXPERIMENT_SIZE,
        'layout_type': 'traditional_aisle',
        'grid_size': width,
        'num_agv': num_agvs,
        'num_tasks': num_tasks,
        'agv_starts': agv_starts,
        'tasks': tasks,
        'obstacles': obstacles,
        'grid': grid,
        'layout_params': {
            'num_vertical_aisles': num_vertical_aisles,
            'num_main_corridors': num_main_corridors,
            'aisle_width': aisle_width,
            'corridor_width': corridor_width
        }
    }

    with open('instance.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(instance, f, allow_unicode=True, default_flow_style=False)

    print(f"\n✓ 地图已保存到 custom_map.yaml")
    print(f"✓ 可视化地图已保存到 {visual_map_file}")
    print(f"✓ 实例配置已保存到 instance.yaml")

    # 打印网格
    print_grid(grid, width, height, agv_starts, tasks)

    # 实验说明
    print("\n" + "=" * 60)
    print("实验说明")
    print("=" * 60)
    if EXPERIMENT_SIZE == 'small':
        print("  用途: 最优性验证")
        print("  方法: 完整MIP vs 枚举法+A*+Q")
        print("  预期: 完整MIP可解，验证启发式方法接近最优")
    elif EXPERIMENT_SIZE == 'medium':
        print("  用途: 中规模测试")
        print("  方法: GA vs Nearest Neighbor + A*+Q")
        print("  预期: 对比GA和贪心法的性能差异")
    else:
        print("  用途: 大规模测试")
        print("  方法: GA + A*+Q")
        print("  预期: 测试GA在大规模问题上的性能极限")

    print(f"\n💡 布局说明: 传统通道式仓库布局")
    print(f"   - 平行货架行 + 垂直拣货通道")
    print(f"   - 水平主通道（{'双向通行' if corridor_width >= 2 else '单向通行'}）")
    print(f"   - 80%+真实仓库采用此标准布局")


if __name__ == "__main__":
    create_instance()
