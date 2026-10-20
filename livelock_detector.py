"""
活锁检测与解除模块

针对 A*+Q 路径规划机制：
- A*+Q 不会产生位置冻结（deadlock）
- 会产生循环震荡（livelock / oscillatory behavior）
- 表现为：AGV持续移动但无法向目标推进
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass
import time


class LivelockType(Enum):
    """活锁类型枚举"""
    OSCILLATION = "oscillation"  # 循环震荡


@dataclass
class LivelockEvent:
    """活锁事件记录"""
    timestep: int
    agents: List[int]
    livelock_type: LivelockType
    resolution_success: bool
    resolution_method: str
    timestamp: float


class LivelockDetector:
    """活锁检测器"""

    def __init__(self, num_agents: int, window_size: int = 12,
                 oscillation_threshold: int = 10, max_livelocks: int = 20):
        """
        初始化活锁检测器

        Args:
            num_agents: AGV数量
            window_size: 位置历史窗口大小
            oscillation_threshold: 震荡判定阈值（连续步数）
            max_livelocks: 最大允许活锁次数
        """
        self.num_agents = num_agents
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        self.max_livelocks = max_livelocks

        self.agents_position_history: Dict[int, List[List[int]]] = {
            i: [] for i in range(num_agents)
        }

        self.livelock_events: List[LivelockEvent] = []
        self.livelock_count = 0

    def record_position(self, agent_id: int, position: List[int], timestep: int):
        """
        记录AGV位置历史

        Args:
            agent_id: AGV ID
            position: 当前位置 [x, y]
            timestep: 时间步
        """
        self.agents_position_history[agent_id].append(position)

        if len(self.agents_position_history[agent_id]) > self.window_size:
            self.agents_position_history[agent_id] = \
                self.agents_position_history[agent_id][-self.window_size:]

    def detect_livelock(self, timestep: int) -> Tuple[bool, List[int], Optional[LivelockType]]:
        """
        综合活锁检测

        A*+Q机制下，只会产生循环震荡活锁，不会产生位置冻结。

        Args:
            timestep: 当前时间步

        Returns:
            (is_livelock, involved_agents, livelock_type)
        """
        if len(self.agents_position_history) < 2:
            return False, [], None

        # 只检测循环震荡
        oscillation_agents = self._detect_oscillation_livelock()
        if len(oscillation_agents) >= 2:
            return True, oscillation_agents, LivelockType.OSCILLATION

        return False, [], None

    def _detect_oscillation_livelock(self) -> List[int]:
        """
        循环震荡活锁检测

        判定条件：
        1. AGV在最近 oscillation_threshold 步内访问的不同位置 <= 3
        2. 存在"来回移动"模式（回退或位移反向）
        3. 至少2个AGV同时震荡

        Returns:
            List[int]: 震荡涉及的AGV列表
        """
        oscillation_agents = []

        for agent, history in self.agents_position_history.items():
            if len(history) < self.oscillation_threshold:
                continue

            recent_positions = history[-self.oscillation_threshold:]

            # 条件1: 不同位置数量 <= 3
            unique_positions = []
            for pos in recent_positions:
                if not any(np.array_equal(pos, up) for up in unique_positions):
                    unique_positions.append(pos)

            if len(unique_positions) > 3:
                continue

            # 条件2: 来回移动模式
            if len(unique_positions) >= 2 and self._is_oscillating_pattern(recent_positions):
                oscillation_agents.append(agent)

        return oscillation_agents

    def _is_oscillating_pattern(self, positions: List[List[int]]) -> bool:
        """
        判断位置序列是否为震荡模式

        A*+Q机制下的震荡特征：
        - 两个AGV在小范围内来回移动
        - 位移向量频繁反向
        - 或位置频繁回退（t步位置 = t-2步位置）

        Args:
            positions: 位置历史列表

        Returns:
            bool: 是否为震荡模式
        """
        if len(positions) < 4:
            return False

        # 方法1: 回退次数检测（t步位置 = t-2步位置）
        backtrack_count = 0
        for i in range(2, len(positions)):
            if np.array_equal(positions[i], positions[i - 2]):
                backtrack_count += 1

        # 回退次数 >= 1/4 判定为震荡
        if backtrack_count >= len(positions) // 4:
            return True

        # 方法2: 位移向量反向检测
        displacements = []
        for i in range(1, len(positions)):
            disp = np.array(positions[i]) - np.array(positions[i - 1])
            if np.any(disp != 0):  # 只记录有位移的
                displacements.append(disp)

        if len(displacements) >= 2:
            reverse_count = 0
            for i in range(1, len(displacements)):
                dot_product = np.dot(displacements[i], displacements[i - 1])
                if dot_product < 0:  # 反向
                    reverse_count += 1

            # 反向次数 >= 2/5 判定为震荡
            if reverse_count >= len(displacements) * 2 // 5:
                return True

        return False

    def should_stop(self) -> bool:
        """检查是否应该停止规划（活锁次数过多）"""
        return self.livelock_count >= self.max_livelocks

    def log_livelock(self, timestep: int, agents: List[int],
                     livelock_type: LivelockType, success: bool,
                     method: str):
        """记录活锁事件"""
        event = LivelockEvent(
            timestep=timestep,
            agents=agents,
            livelock_type=livelock_type,
            resolution_success=success,
            resolution_method=method,
            timestamp=time.time()
        )
        self.livelock_events.append(event)

        if success:
            self.livelock_count += 1

    def reset_history(self, agents: List[int], keep_last: int = 3):
        """
        重置指定AGV的位置历史（活锁解除后使用）

        Args:
            agents: 需要重置的AGV列表
            keep_last: 保留最近的位置数量
        """
        for agent in agents:
            if agent in self.agents_position_history:
                if len(self.agents_position_history[agent]) > keep_last:
                    self.agents_position_history[agent] = \
                        self.agents_position_history[agent][-keep_last:]

    def get_statistics(self) -> Dict:
        """获取活锁统计信息"""
        if not self.livelock_events:
            return {
                'total_livelocks': 0,
                'oscillation_livelocks': 0,
                'successful_resolutions': 0,
                'failed_resolutions': 0,
                'involved_agents': [],
                'timesteps': []
            }

        total = len(self.livelock_events)
        oscillation_count = sum(
            1 for e in self.livelock_events
            if e.livelock_type == LivelockType.OSCILLATION
        )
        success_count = sum(1 for e in self.livelock_events if e.resolution_success)

        involved_agents = set()
        for event in self.livelock_events:
            involved_agents.update(event.agents)

        timesteps = [e.timestep for e in self.livelock_events]

        return {
            'total_livelocks': total,
            'oscillation_livelocks': oscillation_count,
            'successful_resolutions': success_count,
            'failed_resolutions': total - success_count,
            'involved_agents': sorted(list(involved_agents)),
            'timesteps': timesteps
        }

    def print_summary(self):
        """打印活锁统计摘要"""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("活锁检测统计")
        print("=" * 80)
        print(f"  总活锁次数: {stats['total_livelocks']}")
        print(f"    - 循环震荡: {stats['oscillation_livelocks']}")
        print(f"  成功解除: {stats['successful_resolutions']}")
        print(f"  解除失败: {stats['failed_resolutions']}")

        if stats['involved_agents']:
            print(f"  涉及AGV: {stats['involved_agents']}")

        if stats['timesteps']:
            print(f"  发生时间: {stats['timesteps']}")