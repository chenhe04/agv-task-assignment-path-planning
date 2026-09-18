# visualize_paths_gif.py
"""
AGV路径动态可视化模块
用法: python visualize_paths_gif.py
功能: 读取YAML路径文件，生成AGV移动的动态GIF图片
"""

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import argparse
import numpy as np

# ========== 全局配置 ==========
INPUT_PATH_FILE = 'yaml/output_ga_large.yaml'  # ← 在这里修改输入的路径YAML文件
OUTPUT_GIF_FILE = 'agv_paths_large_ga.gif'  # ← 在这里修改输出的GIF文件名
GIF_TITLE = "Large Scale - GA"  # ← 在这里修改GIF标题
ANIMATION_FPS = 1  # ← 在这里修改帧率（1 FPS = 每步1秒）

# 地图文件自动检测
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
        "未找到地图文件！\n"
        "期望的文件: custom_map.yaml 或 map_{small/medium/large}_traditional_aisle.yaml"
    )

MAP_FILE = AUTO_DETECTED_MAP
INSTANCE_FILE = 'instance.yaml'


# ==================================


class AGVPathAnimator:
    """
    AGV路径动态可视化类
    """

    def __init__(self,
                 path_file,
                 map_file=None,
                 instance_file='instance.yaml',
                 title="AGV Path Animation"):
        """
        初始化动画器

        参数:
            path_file: 路径YAML文件路径
            map_file: 地图文件路径（None则使用自动检测）
            instance_file: 实例配置文件路径
            title: GIF标题
        """
        if map_file is None:
            map_file = MAP_FILE

        self.path_file = path_file
        self.map_file = map_file
        self.instance_file = instance_file
        self.title = title

        # 加载数据
        self.load_map()
        self.load_instance()
        self.load_paths()

        # AGV颜色（鲜艳颜色）
        self.colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                       '#1ABC9C', '#E67E22', '#E91E63']

        print(f"\n加载完成:")
        print(f"  地图: {self.width}x{self.height}")
        print(f"  AGV数: {self.num_agvs}")
        print(f"  任务数: {self.num_tasks}")
        print(f"  总时间步: {self.total_steps}")

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

    def load_instance(self):
        """加载测试实例"""
        if not os.path.exists(self.instance_file):
            raise FileNotFoundError(f"实例文件 {self.instance_file} 不存在")

        with open(self.instance_file, 'r') as f:
            self.instance = yaml.load(f, Loader=yaml.Loader)

        self.agv_starts = self.instance['agv_starts']
        self.tasks = self.instance['tasks']
        self.experiment_size = self.instance.get('experiment_size', 'unknown')
        self.num_agvs = len(self.agv_starts)
        self.num_tasks = len(self.tasks)

    def load_paths(self):
        """加载路径数据"""
        if not os.path.exists(self.path_file):
            raise FileNotFoundError(f"路径文件 {self.path_file} 不存在")

        with open(self.path_file, 'r', encoding='utf-8') as f:
            path_data = yaml.load(f, Loader=yaml.Loader)

        self.schedule = path_data['schedule']

        # 获取所有AGV的路径
        self.agent_paths = {}
        max_time = 0

        for agent_key in sorted(self.schedule.keys()):
            agent_id = int(agent_key.replace('agent', ''))
            path = self.schedule[agent_key]

            # 按时间顺序排序
            path_sorted = sorted(path, key=lambda x: x['t'])

            # 提取位置序列（索引即时间步）
            positions = []
            for step in path_sorted:
                t = step['t']
                # 确保列表长度足够
                while len(positions) <= t:
                    if positions:
                        positions.append(positions[-1])  # 保持最后位置
                    else:
                        positions.append((step['x'], step['y']))

                positions[t] = (step['x'], step['y'])
                max_time = max(max_time, t)

            self.agent_paths[agent_id] = positions

        self.total_steps = max_time + 1  # 从0开始计数

    def create_animation(self, save_path='agv_paths.gif', fps=1):
        """
        创建动态GIF

        参数:
            save_path: 输出GIF文件路径
            fps: 帧率（每秒帧数，默认1 FPS = 每步1秒）
        """
        print(f"\n{'=' * 50}")
        print("生成AGV路径动态GIF")
        print(f"{'=' * 50}")
        print(f"  输入文件: {self.path_file}")
        print(f"  输出文件: {save_path}")
        print(f"  帧率: {fps} FPS (每步 {1 / fps:.1f} 秒)")
        print(f"  总帧数: {self.total_steps}")
        print(f"  预计时长: {self.total_steps / fps:.1f} 秒")
        print(f"  标题: {self.title}")

        # 根据地图大小调整图形尺寸
        fig_size = max(10, max(self.width, self.height) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # 设置背景色
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FFFFFF')

        # 绘制静态背景（只绘制一次）
        self.draw_static_background(ax)

        # 初始化AGV标记和时间文本
        agv_markers = []
        for i in range(self.num_agvs):
            color = self.colors[i % len(self.colors)]
            marker, = ax.plot([], [], 'o', markersize=18, color=color,
                              markeredgecolor='white', markeredgewidth=2.5,
                              zorder=10, label=f'AGV{i}')
            agv_markers.append(marker)

        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=13, fontweight='bold',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
                            zorder=15)

        # 添加图例（右上角）
        legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
                           bbox_to_anchor=(0.98, 0.98), ncol=min(self.num_agvs, 4),
                           borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.5)

        def init():
            """初始化动画"""
            for marker in agv_markers:
                marker.set_data([], [])
            time_text.set_text('Time: 0')
            return agv_markers + [time_text]

        def animate(frame):
            """更新每一帧"""
            # 更新每个AGV的位置
            for i in range(self.num_agvs):
                if i in self.agent_paths and frame < len(self.agent_paths[i]):
                    pos = self.agent_paths[i][frame]
                    agv_markers[i].set_data([pos[0]], [pos[1]])
                else:
                    # 如果超出路径长度，保持在最后位置
                    if i in self.agent_paths and len(self.agent_paths[i]) > 0:
                        last_pos = self.agent_paths[i][-1]
                        agv_markers[i].set_data([last_pos[0]], [last_pos[1]])
                    else:
                        agv_markers[i].set_data([], [])

            # 更新时间显示
            time_text.set_text(f'Time: {frame}/{self.total_steps - 1}')

            return agv_markers + [time_text]

        # 创建动画
        print(f"\n正在生成GIF...")
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=self.total_steps, interval=1000 / fps,
                             blit=True, repeat=False)

        # 保存GIF
        writer = PillowWriter(fps=fps, metadata=dict(artist='AGV Simulator'))
        anim.save(save_path, writer=writer, dpi=100)

        print(f"\n✓ GIF已保存: {save_path}")
        print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")

        plt.close()

    def draw_static_background(self, ax):
        """绘制静态背景（障碍物、任务点等）"""
        # 设置坐标轴
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(False)

        # 绘制格子边框
        for x in range(self.width + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, self.height - 0.5],
                    color='#D0D0D0', linewidth=0.3, alpha=0.4, zorder=1)
        for y in range(self.height + 1):
            ax.plot([-0.5, self.width - 0.5], [y - 0.5, y - 0.5],
                    color='#D0D0D0', linewidth=0.3, alpha=0.4, zorder=1)

        # 设置刻度
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.tick_params(labelsize=9)

        # 绘制障碍物
        for obs in self.obstacles:
            ax.add_patch(patches.Rectangle(
                (obs[0] - 0.5, obs[1] - 0.5), 1.0, 1.0,
                facecolor='#666666', edgecolor='#777777', linewidth=0.8,
                zorder=2
            ))

        # 绘制任务点标签
        for i, task in enumerate(self.tasks):
            pickup = task['pickup']
            dropoff = task['dropoff']

            ax.annotate(f'P{i}', (pickup[0], pickup[1]),
                        textcoords="offset points", xytext=(0, 0),
                        fontsize=10, fontweight='bold', ha='center', va='center',
                        color='#FF8C00', zorder=3,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                                  edgecolor='black', alpha=0.6))

            ax.annotate(f'D{i}', (dropoff[0], dropoff[1]),
                        textcoords="offset points", xytext=(0, 0),
                        fontsize=10, fontweight='bold', ha='center', va='center',
                        color='#00CC66', zorder=3,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                                  edgecolor='black', alpha=0.6))

        # 标题（使用自定义标题）
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=10)


def main():
    """主函数"""
    print("=" * 50)
    print("AGV路径动态可视化工具")
    print("=" * 50)

    # 检查文件
    if not os.path.exists(INPUT_PATH_FILE):
        print(f"\n✗ 错误: 路径文件 {INPUT_PATH_FILE} 不存在")
        return

    if not os.path.exists(MAP_FILE):
        print(f"\n✗ 错误: 地图文件 {MAP_FILE} 不存在")
        return

    if not os.path.exists(INSTANCE_FILE):
        print(f"\n✗ 错误: 实例文件 {INSTANCE_FILE} 不存在")
        return

    print(f"\n使用地图: {MAP_FILE}")
    print(f"路径文件: {INPUT_PATH_FILE}")
    print(f"输出文件: {OUTPUT_GIF_FILE}")
    print(f"帧率: {ANIMATION_FPS} FPS")
    print(f"标题: {GIF_TITLE}")

    try:
        # 创建动画器
        animator = AGVPathAnimator(
            path_file=INPUT_PATH_FILE,
            map_file=MAP_FILE,
            instance_file=INSTANCE_FILE,
            title=GIF_TITLE
        )

        # 生成GIF
        animator.create_animation(
            save_path=OUTPUT_GIF_FILE,
            fps=ANIMATION_FPS
        )

        print(f"\n{'=' * 50}")
        print("完成！")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
