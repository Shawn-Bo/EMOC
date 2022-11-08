import datetime
import logging
import os
from typing import Dict


class LoggerHandler(object):
    def __init__(self):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S')

    @staticmethod
    def get_logger_towards_file(name: str, directory: str, filename: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

        logger = logging.getLogger(name)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(os.path.join(directory, filename), 'w', encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def get_logger_basic(name: str):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger = logging.getLogger(name)
        return logger

    def get_loggers_from_args(self, args) -> Dict[str, logging.Logger]:
        logger_dict = {
            "train_logger": None,
            "eval_logger": None,
            "tokenizer_logger": self.get_logger_basic("tokenizer")
        }
        # 确定一个文件夹
        folder_name = self.get_str_year_month_day_hour_minute()
        if args.do_train:
            logger_dict["train_logger"] = loggerHandler.get_logger_towards_file(
                name="train",
                directory=f"./logs/{folder_name}/",
                filename=f"train.log")
        if args.do_eval:
            logger_dict["eval_logger"] = loggerHandler.get_logger_towards_file(
                name="eval",
                directory=f"./logs/{folder_name}/",
                filename=f"eval.log")
        return {name: logger_dict[name] for name in logger_dict if logger_dict[name] is not None}

    @staticmethod
    def get_str_year_month_day_hour_minute():
        return f"{str(datetime.datetime.now().year).zfill(2)}_" \
               f"{str(datetime.datetime.now().month).zfill(2)}_" \
               f"{str(datetime.datetime.now().day).zfill(2)}_" \
               f"{str(datetime.datetime.now().hour).zfill(2)}_" \
               f"{str(datetime.datetime.now().minute).zfill(2)}"

loggerHandler = LoggerHandler()

if __name__ == "__main__":
    pass
