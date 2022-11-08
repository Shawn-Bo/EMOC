import json
import logging
import os
from typing import Dict, Tuple, List, Any, Union

import bidict
import torch
from torch.utils.data import TensorDataset, Subset


class EmotionClassifierDatesetGenerator(object):
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_dataset_from_file(self, abs_path: str, device, tokenizer, label2id: bidict.bidict,
                                   demo_dataset: bool = False,
                                   demo_dataset_ratio: float = 0.0,
                                   max_seq_length: int = 500,
                                   load_labels: bool = True,
                                   expose_source: bool = False) -> Union[TensorDataset, Subset, Any]:
        """
            从课程的训练集中构造张量数据集
        :param load_labels: 加载标签
        :param demo_dataset_ratio: 只加载子集的比例
        :param demo_dataset: 是否只加载子集
        :param device: 加载到的设备
        :param abs_path: 文件的绝对路径
        :param tokenizer: 分词器
        :param label2id: 标签向id的映射
        :param max_seq_length: 最大分词长度
        :return: 构造的张量数据集
        """
        if os.path.exists(abs_path):
            """
                由于数据集的量不大，且不为按行存储，于是直接考虑将其读入内存。
            """
            with open(abs_path, "r", encoding="utf-8") as f:
                data_json_list: List[Dict[str: Any]] = json.load(f)
                """
                [
                  {
                    "id": 1,
                    "content": "天使",
                    "label": "positive"
                  },
                  {
                    "id": 2,
                    "content": "致敬",
                    "label": "positive"
                  }
                ]
                """
                # 获取数据集的样本数，生成数据集
                num_samples = len(data_json_list)
                # 从数据集中构造形式为列表的数据集
                sample_id_list: List[int] = []
                sentence_list: List[str] = []
                if load_labels:
                    label_id_list: List[int] = []
                for data_json in data_json_list:
                    sample_id_list.append(data_json["id"])
                    sentence_list.append(data_json["content"])
                    if load_labels:  # 如果选择加载标签
                        label_id_list.append(label2id[data_json["label"]])
                # 直接通过分词器获取分词，直接返回为张量
                tokenized_tensors: Dict[str, torch.Tensor] = tokenizer(
                    sentence_list,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
                input_ids = tokenized_tensors["input_ids"].to(device)
                token_type_ids = tokenized_tensors["token_type_ids"].to(device)
                attention_mask = tokenized_tensors["attention_mask"].to(device)
                # 构造标签张量（前提是有标签）
                if load_labels:
                    labels = torch.tensor(label_id_list).to(device)
                # 返回最后的数据集
                if load_labels:
                    dataset_full = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
                else:
                    dataset_full = TensorDataset(input_ids, token_type_ids, attention_mask)
                # 原数据集
                source_data = zip(sample_id_list, sentence_list)
                if demo_dataset:
                    # 返回一个子集
                    if expose_source:  # 保留源数据
                        return Subset(dataset_full, range(int(len(dataset_full) * demo_dataset_ratio))), source_data
                    else:
                        return Subset(dataset_full, range(int(len(dataset_full) * demo_dataset_ratio)))
                else:
                    # 返回全量数据集
                    if expose_source:  # 保留原
                        return dataset_full, source_data
                    else:
                        return dataset_full
        else:
            self.logger.error(f"文件路径 '{abs_path}' 不存在！")
