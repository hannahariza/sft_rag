from datasets import load_dataset
from byaldi import RAGMultiModalModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_deepseek import ChatDeepSeek
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import torch, json, os, gc
from rerankers import Reranker
from utils.utils import find_files,pdf_folder_to_images,save_images_to_local,load_png_images,vlm_generate,vlm_generate_multi
from utils.templates import QA_generation_prompt,question_groundedness_critique_prompt,question_standalone_critique_prompt
import logging
from datetime import datetime

# 配置日志
log_dir = "/root/lanyun-tmp/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/qwen25vl_mRAG_eval_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 基础模型路径
BASE_MODEL_PATH = "/root/lanyun-tmp/models_download/qwen2.5vl"
# LoRA适配器路径
LORA_PATH = "/root/lanyun-tmp/results/sft-2"

DATA_PATH = "/root/lanyun-tmp/data_download/mrag/data_new/pdfvqa"
LOG_PATH = "/root/lanyun-tmp/logs"
IMAGE_PATH = "/root/lanyun-tmp/data_download/mrag/data_new/pdfvqa/train_images/train_images"
TMP_PATH = "/root/lanyun-tmp/tmp"
PDF_PATH = "/root/lanyun-tmp/data_download/mrag/pdfs"
ANS_PATH = os.path.join(LOG_PATH, "eval.json") #并没有生成这个文件
DS_API_KEY = "sk-d197234a805846fea250b09bf253923c"
SUBSET = 50
PREPROCESS = 1
INPUT_PDF = -1

if PREPROCESS>0:
    # 加载CSV数据
    import pandas as pd
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train_dataframe.csv"))
    logger.info(f"加载训练数据: {len(train_df)} 条记录")
    
    if SUBSET > 0:
        train_df = train_df.head(SUBSET)
        logger.info(f"使用子集: {len(train_df)} 条记录")
    
    # 转换为HuggingFace数据集格式
    from datasets import Dataset
    dataset = Dataset.from_pandas(train_df)
    
    # 由于图像已经存在，不需要重新保存
    logger.info("图像文件已存在，跳过图像保存步骤")
if INPUT_PDF>0:
    pdf_folder_to_images(input_folder=PDF_PATH)

# 加载图像文件
all_images = load_png_images(IMAGE_PATH)
logger.info(f"加载图像: {len(all_images)} 张")

# 加载基础模型
logger.info(f"开始加载基础模型: {BASE_MODEL_PATH}")
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

# 加载LoRA适配器
logger.info(f"开始加载LoRA适配器: {LORA_PATH}")
vl_model = PeftModel.from_pretrained(vl_model, LORA_PATH)

vl_model.eval()
logger.info("模型加载完成")
min_pixels = 224 * 224
max_pixels = 448 * 448
processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
)   
# print(vlm_generate_multi("描述这张图片","/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/images/image_0.png"))

questions_final = [None] * len(dataset)
answers_final   = [None] * len(dataset)

for image_id, image_data in tqdm(enumerate(dataset), total=len(dataset), desc="preprocessing images"):
    # 新数据集没有'page'字段，需要根据global_id找到对应的图像
    global_id = image_data['global_id']
    question = image_data['question']
    answer = image_data['answer']
    
    # 根据global_id找到对应的图像文件
    # 这里需要根据实际的文件命名规则来匹配
    # 暂时使用第一个图像作为示例
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]
    if image_files:
        image_path = os.path.join(IMAGE_PATH, image_files[image_id % len(image_files)])
        image = Image.open(image_path)
    else:
        logger.warning(f"未找到图像文件，跳过样本 {image_id}")
        continue
    
    output_QA_couples = vlm_generate_multi(vl_model=vl_model,processor=processor,prompt=QA_generation_prompt,img_path=None,image=image,n=7)
    candidates = []
    for output_QA_couple in output_QA_couples:
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            candidates.append((question,answer))
        except:
            continue
    questions,answers = [],[]
    for question,answer in candidates:
        evaluations = {
            "groundedness": vlm_generate(
                vl_model=vl_model,
                processor=processor,
                prompt=question_groundedness_critique_prompt.format(question=question),
                image=image
            )[0],
            "standalone": vlm_generate(
                vl_model=vl_model,
                processor=processor,
                prompt=question_standalone_critique_prompt.format(question=question),
                image=image
            )[0],
        }
        # print(evaluations)
        try:
            scores = []
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                scores.append(score)
        except Exception as e:
            continue
        # print("score:",sum(scores)/len(scores))
        if sum(scores)/len(scores)>=4:
            questions.append(question)
            answers.append(answer)
            
    questions_final[image_id] = questions
    answers_final[image_id] = answers
    # print(len(answers))
    
dataset = dataset.remove_columns(
    [c for c in ["questions", "answers"] if c in dataset.column_names]
)
dataset = dataset.add_column("questions", questions_final)
dataset = dataset.add_column("answers",   answers_final)
df = dataset.to_pandas()
output_path = "/root/lanyun-tmp/data_download/mrag/mypdfqa/pdf_qa.parquet"
df.to_parquet(output_path, index=False)
    
        