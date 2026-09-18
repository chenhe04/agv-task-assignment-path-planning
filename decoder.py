import yaml
from planning import Planning
import numpy as np


class Decoder():
    def __init__(self, matrix_order, tasks=None, map_file=None):
        """
        参数:
            matrix_order: 任务执行顺序
            tasks: 任务列表 [{'pickup':[x,y], 'dropoff':[x,y]}, ...]
            map_file: 地图文件路径（可选）
        """
        if isinstance(matrix_order[0], list):
            matrix_order = [item[0] if isinstance(item, list) else item for item in matrix_order]
        self.matrix_order = matrix_order
        self.tasks = tasks  # 保存传入的任务
        self.map_file = map_file if map_file else 'map_normal_congestion_low_compare.yaml'

    def main(self):
        # 如果传入了任务，直接使用传入的任务
        if self.tasks is not None:
            tasks_data = {}
            for i, task in enumerate(self.tasks):
                tasks_data[i] = {
                    "point1": [task['pickup'][1], task['pickup'][0]],
                    "point2": [task['dropoff'][1], task['dropoff'][0]],
                }
        else:
            # 否则从地图文件读取
            with open(self.map_file, 'r', encoding='utf-8') as f:
                map_data = yaml.load(f, Loader=yaml.Loader)

            initial_tasks = map_data['tasks']['initial']
            tasks_data = {}

            for i, task in enumerate(initial_tasks):
                tasks_data[i] = {
                    "point1": [task['pickup'][1], task['pickup'][0]],
                    "point2": [task['dropoff'][1], task['dropoff'][0]],
                }

        # matrix_orderの順番に並び替え
        sorted_data = {}
        for i in range(len(self.matrix_order)):
            sorted_data[i] = tasks_data[self.matrix_order[i]]

        output_data = []
        for i in range(len(self.matrix_order)):
            output_data.append({
                "name": "task%d" % i,
                "point1": sorted_data[i]["point1"],
                "point2": sorted_data[i]["point2"]
            })
        output = {"tasks3": output_data}

        planning = Planning(task_dict=output, disturbance=[])
        step, elapsed_time, dist = planning.main()
        if step == 0 and elapsed_time == 0:
            step = np.inf

        return step


class Decoder_Incremental():
    def __init__(self, new_task_assignment, new_tasks, existing_paths, agv_completion_info, map_file=None):
        self.new_task_assignment = new_task_assignment
        self.new_tasks = new_tasks
        self.existing_paths = existing_paths
        self.agv_completion_info = agv_completion_info
        self.map_file = map_file if map_file else 'map_normal_congestion_low_compare.yaml'

    def main(self):
        with open(self.map_file, 'r', encoding='utf-8') as f:
            map_data = yaml.load(f, Loader=yaml.Loader)

        conv_paths = {}
        for agv_id, path in self.existing_paths.items():
            conv_paths[agv_id] = [[p[1], p[0]] for p in path]

        output_data = []
        task_idx_map = []

        for agv_id, tasks in enumerate(self.new_task_assignment):
            idx_list = []
            for t in tasks:
                output_data.append({
                    "name": "task%d" % len(output_data),
                    "point1": [t['pickup'][1], t['pickup'][0]],
                    "point2": [t['dropoff'][1], t['dropoff'][0]]
                })
                idx_list.append(len(output_data) - 1)
            task_idx_map.append(idx_list)

        output = {"tasks3": output_data}

        from planning import Planning_Incremental
        planning = Planning_Incremental(
            task_dict=output, disturbance=[],
            existing_paths=conv_paths,
            agv_completion_info=self.agv_completion_info,
            new_task_assignment=task_idx_map
        )

        step, elapsed_time, dist, additional_task_paths = planning.main()
        if step == 0 and elapsed_time == 0:
            step = np.inf

        self.additional_task_paths = additional_task_paths
        return step