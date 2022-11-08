import argparse
import json
import logging
import os.path
import random

"""
    只用于参数解析，不影响任何执行逻辑。
"""


class Args(object):
    def __str__(self):
        return str(self.__dict__)


class CLIParser(object):
    __slots__ = ('parser', 'logger')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_args()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def set_args(self):
        self.parser.add_argument("--model", default=None, type=str)
        self.parser.add_argument("--output_dir", default="./output/", type=str)
        self.parser.add_argument("--eval_per_epoch", default=10, type=int,
                                 help="How many times it evaluates on dev set per epoch")
        self.parser.add_argument("--max_seq_length", default=500, type=int,
                                 help="The maximum total input sequence length after WordPiece tokenization. \n"
                                      "Sequences longer than this will be truncated, and sequences shorter \n"
                                      "than this will be padded.")
        self.parser.add_argument("--negative_label", default="no_relation", type=str)
        self.parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.")
        self.parser.add_argument("--train_file", default=None, type=str, help="The path of the training benchmarks.")
        self.parser.add_argument("--train_mode", type=str, default='random_sorted',
                                 choices=['random', 'sorted', 'random_sorted'])
        self.parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        self.parser.add_argument("--do_lower_case", action='store_true',
                                 help="Set this flag if you are using an uncased model.")
        self.parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test.txt set.")
        self.parser.add_argument("--eval_with_gold", action="store_true",
                                 help="Whether to evaluate the relation model with gold entities provided.")
        self.parser.add_argument("--train_batch_size", default=32, type=int,
                                 help="Total batch size for training.")
        self.parser.add_argument("--eval_batch_size", default=8, type=int,
                                 help="Total batch size for eval.")
        self.parser.add_argument("--eval_metric", default="f1", type=str)
        self.parser.add_argument("--learning_rate", default=None, type=float,
                                 help="The initial learning rate for Adam.")
        self.parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                 help="Total number of training epochs to perform.")
        self.parser.add_argument("--warmup_proportion", default=0.1, type=float,
                                 help="Proportion of training to perform linear learning rate warmup for. "
                                      "E.g., 0.1 = 10%% of training.")
        self.parser.add_argument("--no_cuda", action='store_true',
                                 help="Whether not to use CUDA when available")
        self.parser.add_argument('--seed', type=int, default=random.randint(0, 100),
                                 help="random seed for initialization")
        self.parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

        self.parser.add_argument("--entity_output_dir", type=str, default=None,
                                 help="The directory of the prediction files of the entity model")
        self.parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json",
                                 help="The entity prediction file of the dev set")
        self.parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json",
                                 help="The entity prediction file of the test.txt set")

        self.parser.add_argument("--prediction_file", type=str, default="predictions.json",
                                 help="The prediction filename for the relation model")

        self.parser.add_argument('--task', type=str, default=None,
                                 choices=['ace04', 'ace05', 'scierc', 'cmeie', 'diakg'])
        self.parser.add_argument('--context_window', type=int, default=0)

        self.parser.add_argument('--add_new_tokens', action='store_true',
                                 help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
        self.parser.add_argument('--batch_computation', action='store_true',
                                 help="Whether to use batch computation to speedup the inference.")
        self.parser.add_argument("--cuda_visible_devices", type=str, default="0")
        self.parser.add_argument("--cpu", action="store_true")

    def parse_args(self, display_log=False, save_config_file: str = "", load_config_file: str = ""):
        if not save_config_file == "" and not load_config_file == "":
            """
                不可能同时加载并保存同一个参数文件。
            """
            raise FileNotFoundError("arguments save_config_file and load_config_file can't be both True.")
        elif load_config_file == "":
            """
                从命令行中解析参数，通过argParser的方式。
                如果save_config_file，则保存参数到指定位置。
            """
            args = self.parser.parse_args()
            if display_log:
                self.logger.info(args)
            if not save_config_file == "":
                """
                    将解析的参数保存至相应目录中。
                """
                args_dict = args.__dict__
                with open(save_config_file, "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=2)
                if display_log:
                    self.logger.info(f"args saved in {save_config_file}")
            return args
        else:
            """
                自文件中读取参数，并不使用特殊的读取工具，只是通过修改一个空类的__dict__字段
            """
            if not os.path.exists(load_config_file):
                raise FileNotFoundError(f"CLI config file '{load_config_file}' not exist.")
            # 从文件中读取参数到dict
            with open(load_config_file, "r", encoding="utf-8") as f:
                args_dict = json.load(f)
            # 构造args
            args = Args()
            args.__dict__ = args_dict
            return args

    @staticmethod
    def check_args(args):
        """
            只对输入参数的正确性进行检查，不改变程序和工程状态。

            :param args: 解析的参数
            :return:
        """
        # if not args.do_train and not args.do_eval:
        #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")
        pass


argParser = CLIParser()

if __name__ == "__main__":  # 测试
    print(argParser.parse_args(display_log=True, load_config_file=r"D:\PrivateProjects\EMOC\configs\train.config"))
