import json
import logging
import os


class NRelationCounter(object):
    __slots__ = "logger", "state"

    class CounterState(object):
        __slots__ = "count_rel"

        def __init__(self):
            self.count_rel = 0

    def __init__(self, name=None):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__ if name is None else name)
        self.state = NRelationCounter.CounterState()

    def initialize_func(self):
        self.state = NRelationCounter.CounterState()

    def extrapolate_func(self, snap: dict):
        self.state.count_rel += len(snap["relations"][0])

    def summarize_func(self):
        # 这个函数只用于状态的转换，不用于数据的产生
        pass

    def count_rel_from_file(self, abs_path: str):
        self.initialize_func()
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                for line_text in f:
                    line_json = json.loads(line_text)
                    self.extrapolate_func(line_json)
            self.summarize_func()
            return self.state.count_rel
        else:
            self.logger.error(f"文件路径 '{abs_path}' 不存在！")
