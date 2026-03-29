import torch
import logging
import os
from datetime import datetime
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

# 配置日志
log_dir = "/root/lanyun-tmp/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/qwen25vl_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            {"type": "text", "text": "介绍一下colpali的原理"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    # videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# import IPython;IPython.embed()
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
logger.info(f"模型输出: {output_text}")
print(output_text)
