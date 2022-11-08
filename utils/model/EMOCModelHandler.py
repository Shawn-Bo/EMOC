import datetime
import logging
import os
from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

from utils.common.Mappings import get_bidict_from_list


class EMOCModelHandler(object):
    """
        管理模型相关的一切草走
    """

    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def load_model(self, model: str, load_model_bin=False):
        """
            加载并返回模型，根据参数决定是否使用模型
        :param model:
        :param load_model_bin:
        :return:
        """
        if load_model_bin:
            # 此时，model表示模型权重文件路径
            self.logger.info(f"load model from weighted binary file: {model}.")
            return torch.load(model)
        else:
            # 此时，model表示transformers中的模型名称
            self.logger.info(f"load model from scratch or checkpoint: {model}.")
            return AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)

    def get_processor(self, tokenizer_model: str, task_emotion_classifier_labels: List[str],
                      max_seq_length: int = 500):
        self.logger.info(f"tokenizer_model: {tokenizer_model}, "
                         f"task_emotion_classifier_labels: {task_emotion_classifier_labels},"
                         f"max_seq_length: {max_seq_length}")
        return EMOCIOProcessor(tokenizer_model, task_emotion_classifier_labels, max_seq_length)


class EMOCCheckpointSaver(object):
    def __init__(self, checkpoint_root_abspath: str):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.folder_path = f"{checkpoint_root_abspath}/{self.get_str_year_month_day_hour_minute()}"
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

    def save_checkpoint(self, model, model_save_name: str):
        """
            将模型保存到目标位置。
        :return:
        """
        save_path = f"{self.folder_path}/{model_save_name}"
        with open(save_path, "wb") as f:
            torch.save(model, f)
            self.logger.info(f"MODEL WEIGHT SAVED TO {save_path}.")

    @staticmethod
    def get_str_year_month_day_hour_minute():
        return f"{str(datetime.datetime.now().year).zfill(2)}_" \
               f"{str(datetime.datetime.now().month).zfill(2)}_" \
               f"{str(datetime.datetime.now().day).zfill(2)}_" \
               f"{str(datetime.datetime.now().hour).zfill(2)}_" \
               f"{str(datetime.datetime.now().minute).zfill(2)}"



class EMOCIOProcessor(object):
    """
        处理器，负责管理处理测数据的工具，以及从输入中进行预测。
    """

    def __init__(self, tokenizer_model: str, task_emotion_classifier_labels: List[str],
                 max_seq_length: int = 500):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.bidict_label2id = get_bidict_from_list(task_emotion_classifier_labels).inverse
        self.max_length = max_seq_length

    def preprocess(self, sentence_list: List[str]):
        # 从输入中构建训练数据，可以直接输入到网络中。
        # 通过分词器获取分词，返回为张量
        tokenized_tensors: Dict[str, torch.Tensor] = self.tokenizer(
            sentence_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokenized_tensors["input_ids"]
        token_type_ids = tokenized_tensors["token_type_ids"]
        attention_mask = tokenized_tensors["attention_mask"]
        return input_ids, token_type_ids, attention_mask

    def postprocessor(self, model_outputs):
        """
            从预测出来的张量中生成可以对应的预测数据
        :return:
        """
        predicted_label_ids = F.softmax(model_outputs.logits, dim=1).argmax(dim=-1)
        predicted_label = [self.bidict_label2id.inverse[predicted_label_id]
                           for predicted_label_id in predicted_label_ids.to(int).tolist()]
        return predicted_label

    def get_bidict_label2id(self):
        return self.bidict_label2id

    def get_tokenizer(self):
        """
            获取预处理器的分词器。
        :return: 分词器
        """
        return self.tokenizer


if __name__ == "__main__":
    modelHandler = EMOCModelHandler()
    # 预处理器
    processor = modelHandler.get_processor(
        tokenizer_model="bert-base-chinese",
        task_emotion_classifier_labels=["neutral", "positive", "negative"]  # 原则：不藏着
    )
    input_ids, token_type_ids, attention_mask = processor.preprocess(["我真是吐了啊", "什么伞兵玩意"])
    # 模型
    model = modelHandler.load_model(
        model="bert-base-chinese"
    )
    model_outputs = model(input_ids, token_type_ids, attention_mask)
    # 后处理器
    results = processor.postprocessor(model_outputs)
    print(results)
