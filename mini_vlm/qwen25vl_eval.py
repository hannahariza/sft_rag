import torch
import logging
import os
from datetime import datetime
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from utils.utils import find_files,format_data_chartqa
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

# 配置日志
log_dir = "/root/lanyun-tmp/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/qwen25vl_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_APTH = "/root/lanyun-tmp/results/sft-2"
DATA_PATH = "/root/lanyun-tmp/data_download/sft"
TMP_PATH = "/root/lanyun-tmp/tmp"
SUBSET = -1

logger.info("开始加载数据集")
directories = ['data']
data_files = find_files(directories,DATA_PATH)[:1]
logger.info(f"找到数据文件: {len(data_files)} 个")
dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
if SUBSET > 0:
    train_dataset = dataset.select(range(SUBSET))
    logger.info(f"使用子集: {SUBSET} 个样本")

train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
train_dataset, eval_dataset = train_val_dataset.train_test_split(test_size=0.1, seed=42).values()
logger.info(f"数据集分割完成 - 训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}, 测试集: {len(test_dataset)}")

logger.info("开始格式化数据集")
train_dataset = [format_data_chartqa(sample) for sample in train_dataset]
eval_dataset = [format_data_chartqa(sample) for sample in eval_dataset]
test_dataset = [format_data_chartqa(sample) for sample in test_dataset]
logger.info("数据集格式化完成")

# 基础模型路径
BASE_MODEL_PATH = "/root/lanyun-tmp/models_download/qwen2.5vl"
# LoRA适配器路径
LORA_PATH = "/root/lanyun-tmp/results/sft-2"

logger.info(f"开始加载基础模型: {BASE_MODEL_PATH}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

logger.info(f"开始加载LoRA适配器: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
logger.info("模型加载完成")

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    logger.info(f"处理样本: {sample}")
    # 确保logs目录存在
    os.makedirs("logs", exist_ok=True)
    sample[1]['content'][0]['image'].save("logs/eval_test.png")
    text_input = processor.apply_chat_template(
        sample[:2],
        tokenize=False,
        add_generation_prompt=True
    )
    logger.info(f"文本输入: {text_input}")
    print("text_input",text_input)
    image_inputs, _ = process_vision_info(sample)
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]  

output = generate_text_from_sample(model, processor, test_dataset[0])
logger.info(f"模型输出: {output}")
print(output)
import IPython;IPython.embed()