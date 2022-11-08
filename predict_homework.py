import json

from utils.model.EMOCModelHandler import EMOCModelHandler
from utils.preheat.CLIParser import argParser
from utils.preheat.DeviceHandler import deviceHandler
from utils.preheat.SeedHandler import seedHandler

args = argParser.parse_args(
    display_log=True,
    load_config_file="configs/train.config"
)
seedHandler.set_seed(args.seed)
modelHandler = EMOCModelHandler()
device = deviceHandler.get_device(args.cuda_visible_devices, args.cpu, display_log=True)
# 预处理器
processor = modelHandler.get_processor(
    tokenizer_model=args.tokenizer_model,
    task_emotion_classifier_labels=["neutral", "positive", "negative"],
    max_seq_length=args.max_seq_length
)
# 模型
model = modelHandler.load_model(model=args.checkpoint_weight, load_model_bin=args.load_model_bin)
model.to(device)

# 楷书处理数据
with open("benchmarks/test.json", "r", encoding="utf-8") as f:
    test_json = json.load(f)
sentence_texts = [json_item["content"] for json_item in test_json]
sentence_labels = []
ids = []
for json_item in test_json:
    input_ids, token_type_ids, attention_mask = processor.preprocess([json_item["content"]])
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    results = processor.postprocessor(model_outputs)
    sentence_labels.extend(results)
    ids.append(json_item["id"])
    print(results[0], json_item["content"])

# 保存了保存了
label2id = {"neutral": 0, "positive": 1, "negative": 2}
with open("answer.csv", "w") as f:
    for id, sentence_label in zip(ids, sentence_labels):
        f.write(f"{id},{label2id[sentence_label]}\n")
