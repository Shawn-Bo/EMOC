import os

from torch.optim import SGD, AdamW
from tqdm.auto import tqdm

from utils.model.EMOCModelHandler import EMOCModelHandler, EMOCCheckpointSaver

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import bidict
import evaluate
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from utils.common.Mappings import get_bidict_from_list
from utils.iterative.Generator import EmotionClassifierDatesetGenerator
from utils.preheat.CLIParser import argParser
from utils.preheat.DeviceHandler import deviceHandler
from utils.preheat.LoggerHandler import loggerHandler
from utils.preheat.SeedHandler import seedHandler


def project_path(relative_path: str):
    """
        用于从相对路径获取绝对路径，个人感觉没大用。
    :param relative_path: 相对路径
    :return:
    """
    return os.path.dirname(os.path.realpath(__file__)) + "/" + relative_path


def main(args):
    """
        环境准备
        1. 获取设备信息
        2. 设置随机种子
        3. 获取日志器
    """
    # 获取设备信息
    device = deviceHandler.get_device(args.cuda_visible_devices, args.cpu, display_log=True)
    # 设定随机种子
    seedHandler.set_seed(args.seed)
    # 获取日志器
    loggers = loggerHandler.get_loggers_from_args(args)
    train_logger = loggers["train_logger"]
    eval_logger = loggers["eval_logger"]
    tokenizer_logger = loggers["tokenizer_logger"]
    """
        1. 加载数据
            1.1 加载训练数据
            1.1 加载测试数据
        2. 设置优化器
        3. 设置模型
        4. 设置损失函数
        5. 训练主循环
            5.1 批量加载数据
            5.2 前向
            5.3 算loss
            5.4 后向
            5.5 更新
            5.6 测试情况
    """
    modelHandler = EMOCModelHandler()
    processor = modelHandler.get_processor(
        tokenizer_model=args.model,
        task_emotion_classifier_labels=["neutral", "positive", "negative"])

    label2id: bidict.bidict = processor.get_bidict_label2id()
    tokenizer = processor.get_tokenizer()
    # 加载训练数据
    train_dataloader = DataLoader(
        EmotionClassifierDatesetGenerator().generate_dataset_from_file(
            project_path("benchmarks/train_data.json"),
            device,
            tokenizer,
            label2id,
            demo_dataset=args.demo_dataset,
            demo_dataset_ratio=args.demo_dataset_ratio,
            max_seq_length=args.max_seq_length),
        batch_size=args.train_batch_size)
    # 加载测试数据
    eval_dataset = EmotionClassifierDatesetGenerator().generate_dataset_from_file(
            project_path("benchmarks/eval_data.json"),
            device,
            tokenizer,
            label2id,
            demo_dataset=args.demo_dataset,
            demo_dataset_ratio=args.demo_dataset_ratio,
            max_seq_length=args.max_seq_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.train_batch_size)
    """
        加载模型，这里使用bert-base-chinese
    """
    model = modelHandler.load_model(model=args.checkpoint_weight, load_model_bin=args.load_model_bin)
    model.to(device)
    """
        优化器和训练计划
    """
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_epoch = args.train_epoch
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=train_epoch * len(train_dataloader)
    # )
    f1_metric = evaluate.load("f1")
    best_f1 = -1
    checkpointSaver = EMOCCheckpointSaver(project_path("checkpoints"))
    """
        训练主循环
    """
    for epoch in range(train_epoch):
        epoch_total_loss = 0
        epoch_total_step = 0
        model.train()
        for (input_ids, token_type_ids, attention_mask, labels) in tqdm(train_dataloader):
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            epoch_total_loss += loss  # 其实是和下面的约分了
            epoch_total_step += 1
        # 打印出一个epoch的平均loss
        print(f"loss at epoch {epoch}: {epoch_total_loss/epoch_total_step}")
        """
            每个epoch，在eval集上评估模型性能
        """
        model.eval()
        for input_ids, token_type_ids, attention_mask, labels in eval_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            predicted_label = F.softmax(outputs.logits, dim=1).argmax(dim=-1)
            f1_metric.add_batch(references=labels.tolist(), predictions=predicted_label.tolist())
        f1 = f1_metric.compute(average="macro")["f1"]
        eval_logger.info(f"at epoch {epoch}: f1 = {f1}")
        if f1 > best_f1:
            train_logger.info(f"at epoch {epoch}: best f1 = {f1}")
            best_f1 = f1
            # 保存模型即可
            checkpointSaver.save_checkpoint(model, f"model_weight_epoch_{epoch}_f1_{int(f1*100)}.bin")

        """
            保存验证集上最好的模型，作为最终的训练结果。
        """


if __name__ == "__main__":
    main(
        argParser.parse_args(
            display_log=True,
            load_config_file="configs/train.config"
        )
    )
