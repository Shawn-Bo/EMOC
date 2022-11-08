import json
import logging
import os.path
from typing import Any


class FileReader(object):
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def load_json_single_line(self, path: str, encoding="utf-8") -> Any:
        if os.path.exists(path):
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        else:
            self.logger.error(f"文件路径 '{path}' 不存在！")


fileReader = FileReader()

if __name__ == "__main__":
    pass
