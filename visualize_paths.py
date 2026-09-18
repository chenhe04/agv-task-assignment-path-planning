# visualize_paths.py
"""
仓库布局可视化模块
用法: python visualize_paths.py
功能: 读取 instance.yaml 和地图文件，生成仓库布局静态图片
      显示障碍物、AGV起点、取货点和送货点位置
"""

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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

AUTO_DETECTED_MAP = None
for map_file in CUSTOM_MAP_PRIORITY:
    if os.path.exists(map_file):
        AUTO_DETECTED_MAP = map_file
        break

if AUTO_DETECTED_MAP is None:
    raise FileNotFoundError(
        "未找到地图文件！请先运行 step1_create_instance.py 生成地图。\n"
        "期望的文件: custom_map.yaml 或 map_{small/medium/large}_traditional_aisle.yaml"
    )

# 最终使用的地图文件
MAP_FILE = AUTO_DETECTED_MAP


# ================================================


class WarehouseLayoutVisualizer:
    """
    仓库布局可视化类（静态展示）
    """

    def __init__(self,
                 map_file=None,
                 instance_file='instance.yaml'):
        """
        初始化可视化器

        参数:
            map_file: 地图文件路径（None则使用自动检测的地图）
            instance_file: 实例配置文件路径
        """
        # 如果未指定地图文件，使用自动检测的结果
        if map_file is None:
            map_file = MAP_FILE

        self.map_file = map_file
        self.instance_file = instance_file

        # 加载数据
        self.load_map()
        self.load_instance()

        # AGV颜色（鲜艳颜色）
        self.colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#E91E63']

    def load_map(self):
        """加载地图和障碍物"""
        with open(self.map_file, 'r', encoding='utf-8') as f:
            map_data = yaml.load(f, Loader=yaml.Loader)

        self.width = map_data['map']['width']
        self.height = map_data['map']['height']

        # 提取障碍物坐标
        self.obstacles = []
        for y in range(self.height):
            for x in range(self.width):
                if map_data['map']['data'][y][x] == 1:
                    self.obstacles.append([x, y])

        print(f"地图: {self.width}x{self.height}, 障碍物数: {len(self.obstacles)}")

    def load_instance(self):
        """加载测试实例"""
        if not os.path.exists(self.instance_file):
            raise FileNotFoundError(f"实例文件 {self.instance_file} 不存在")

        with open(self.instance_file, 'r') as f:
            self.instance = yaml.load(f, Loader=yaml.Loader)

        self.agv_starts = self.instance['agv_starts']
        self.tasks = self.instance['tasks']

        # 获取实验规模信息
        self.experiment_size = self.instance.get('experiment_size', 'unknown')
        self.num_agvs = len(self.agv_starts)
        self.num_tasks = len(self.tasks)

        print(f"实例: {self.experiment_size}规模, {self.num_agvs}台AGV, {self.num_tasks}个任务")

    def draw_layout(self, ax):
        """绘制仓库布局（障碍物、起点、任务点）"""

        # 设置坐标轴范围（确保整数坐标在格子中心）
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')

        # 隐藏默认的网格线
        ax.grid(False)

        # 手动绘制格子边框（非常浅的灰色，不干扰视觉）
        for x in range(self.width + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, self.height - 0.5],
                    color='#D0D0D0', linewidth=0.3, alpha=0.4, zorder=1)
        for y in range(self.height + 1):
            ax.plot([-0.5, self.width - 0.5], [y - 0.5, y - 0.5],
                    color='#D0D0D0', linewidth=0.3, alpha=0.4, zorder=1)

        # 设置刻度标签（显示整数，但实际位置在格子中心）
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.tick_params(labelsize=9)

        # 绘制障碍物（深灰色方块，填满整个格子）
        for obs in self.obstacles:
            ax.add_patch(patches.Rectangle(
                (obs[0] - 0.5, obs[1] - 0.5), 1.0, 1.0,
                facecolor='#666666', edgecolor='#777777', linewidth=0.8,
                zorder=2
            ))

        # 绘制AGV起点（仅文字标签，居中显示）
        for i, start in enumerate(self.agv_starts):
            if i < len(self.colors):
                # 添加文字标签（对应颜色 + 黑色描边）
                ax.annotate(f'AGV{i}', (start[0], start[1]),
                            textcoords="offset points", xytext=(0, 0),
                            fontsize=11, fontweight='bold', ha='center', va='center',
                            color=self.colors[i], zorder=6,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                                      edgecolor='black', alpha=0.8))

        # 绘制任务点（仅文字标签，居中显示）
        for i, task in enumerate(self.tasks):
            pickup = task['pickup']
            dropoff = task['dropoff']

            # P标签（橙色 + 黑色描边）
            ax.annotate(f'P{i}', (pickup[0], pickup[1]),
                        textcoords="offset points", xytext=(0, 0),
                        fontsize=11, fontweight='bold', ha='center', va='center',
                        color='#FF8C00', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                                  edgecolor='black', alpha=0.8))

            # D标签（绿色 + 黑色描边）
            ax.annotate(f'D{i}', (dropoff[0], dropoff[1]),
                        textcoords="offset points", xytext=(0, 0),
                        fontsize=11, fontweight='bold', ha='center', va='center',
                        color='#00CC66', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                                  edgecolor='black', alpha=0.8))

        # 设置标题
        title = f"Warehouse Layout - {self.experiment_size.upper()} Scale\n" \
                f"{self.width}×{self.height} Grid, {self.num_agvs} AGVs, {self.num_tasks} Tasks"
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    def visualize(self, save_path='warehouse_layout.png'):
        """生成仓库布局可视化图片"""

        print("\n" + "=" * 50)
        print("生成仓库布局可视化图片")
        print("=" * 50)

        # 根据地图大小调整图形尺寸
        fig_size = max(10, max(self.width, self.height) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # 设置背景色
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FFFFFF')

        self.draw_layout(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"\n✓ 已保存: {save_path}")
        plt.show()


def main():
    """主函数"""
    print("=" * 50)
    print("仓库布局可视化工具")
    print("=" * 50)

    print(f"\n使用地图: {MAP_FILE}")
    print(f"实例配置: instance.yaml")

    # 检查地图文件
    if not os.path.exists(MAP_FILE):
        print(f"  ✗ 地图文件 {MAP_FILE} 不存在")
        return

    # 检查实例文件
    if not os.path.exists('instance.yaml'):
        print(f"  ✗ 实例文件 instance.yaml 不存在")
        return

    print(f"  ✓ 地图文件存在")
    print(f"  ✓ 实例文件存在")

    # 创建可视化器
    visualizer = WarehouseLayoutVisualizer(
        map_file=MAP_FILE,
        instance_file='instance.yaml'
    )

    # 生成可视化图片
    visualizer.visualize(save_path='warehouse_layout.png')


if __name__ == "__main__":
    main()
