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
import torch, json, os, gc, logging
from datetime import datetime
from PIL import Image
from rerankers import Reranker
from utils.utils import find_files,pdf_folder_to_images,clear_memory,save_images_to_local_wo_resize,save_images_to_local,load_png_images,images_to_base64,get_grouped_images,process_ranker_results

# 配置日志
log_dir = "/root/lanyun-tmp/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/qwen25vl_mRAG_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 基础模型路径
BASE_MODEL_PATH = "/root/lanyun-tmp/models_download/qwen2.5vl"
# LoRA适配器路径
LORA_PATH = "/root/lanyun-tmp/results/sft-2"
PDF_PATH = "/root/lanyun-tmp/data_download/pdf"
RERANK_MODEL_PATH = "/root/lanyun-tmp/models_download/MonoQwen2-VL-v0.1" #重新排序模型路径
RETRIEVAL_MODEL_PATH = "/root/lanyun-tmp/models_download/colqwen2-v1.0" #检索模型路径
# 新数据集路径
DATA_PATH = "/root/lanyun-tmp/data_download/mrag/data_new/pdfvqa"
LOG_PATH = "/root/lanyun-tmp/logs"
IMAGE_PATH = "/root/lanyun-tmp/data_download/mrag/data_new/pdfvqa/train_images/train_images"
TMP_PATH = "/root/lanyun-tmp/tmp"
# 检索结果保存路径
RETRIEVAL_BEFORE_RERANK_PATH = "/root/lanyun-tmp/tmp/retrieval_before_rerank"
RETRIEVAL_AFTER_RERANK_PATH = "/root/lanyun-tmp/tmp/retrieval_after_rerank"
ANS_PATH = os.path.join(LOG_PATH, "eval.json")
DS_API_KEY = "sk-f7e3dab22ef84202bd9f9f3f276776b5"  # 设置为None跳过DeepSeek评估（API余额不足）
SUBSET = 50
PREPROCESS = 1
INPUT_PDF = -1

if PREPROCESS>0:
    # 加载生成的问答对数据
    import pandas as pd
    qa_file = "/root/lanyun-tmp/data_download/mrag/mypdfqa/pdf_qa.parquet"
    if os.path.exists(qa_file):
        qa_df = pd.read_parquet(qa_file)
        logger.info(f"加载问答对数据: {len(qa_df)} 条记录")
        
        if SUBSET > 0:
            qa_df = qa_df.head(SUBSET)
            logger.info(f"使用子集: {len(qa_df)} 条记录")
        
        # 转换为HuggingFace数据集格式
        from datasets import Dataset
        dataset = Dataset.from_pandas(qa_df)
        logger.info("问答对数据加载完成")
    else:
        logger.error(f"问答对文件不存在: {qa_file}")
        exit(1)
if INPUT_PDF>0:
    pdf_folder_to_images(input_folder=PDF_PATH)

# 检查图像路径是否存在
if not os.path.exists(IMAGE_PATH):
    logger.error(f"图像路径不存在: {IMAGE_PATH}")
    exit(1)

all_images = load_png_images(IMAGE_PATH)
logger.info(f"加载了 {len(all_images)} 张图像")

# 使用离线模式加载检索模型
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    logger.info(f"开始加载检索模型: {RETRIEVAL_MODEL_PATH}")
    if os.path.exists(RETRIEVAL_MODEL_PATH):
        retrieval_model = RAGMultiModalModel.from_pretrained(RETRIEVAL_MODEL_PATH)
        retrieval_model.index(input_path=IMAGE_PATH, index_name="paper_index", store_collection_with_index=False, overwrite=True)
        logger.info("检索模型加载完成")
    else:
        logger.warning(f"检索模型路径不存在: {RETRIEVAL_MODEL_PATH}")
        retrieval_model = None
except Exception as e:
    logger.error(f"检索模型加载失败: {e}")
    logger.info("跳过检索模型，使用简化模式")
    retrieval_model = None

try:
    logger.info(f"开始加载重排序模型: {RERANK_MODEL_PATH}")
    if os.path.exists(RERANK_MODEL_PATH):
        # 使用本地Qwen2-VL-2B-Instruct模型作为processor，避免从huggingface下载
        processor_path = "/root/lanyun-tmp/models_download/Qwen2-VL-2B-Instruct"
        reranker_model = Reranker(
            RERANK_MODEL_PATH, 
            device="cuda",
            processor_name=processor_path,
            attention_implementation="eager"  # 禁用FlashAttention2
        )
        logger.info("重排序模型加载完成")
    else:
        logger.warning(f"重排序模型路径不存在: {RERANK_MODEL_PATH}")
        reranker_model = None
except Exception as e:
    logger.error(f"重排序模型加载失败: {e}")
    logger.info("跳过重排序模型，使用简化模式")
    reranker_model = None

# 加载基础模型
logger.info(f"开始加载基础模型: {BASE_MODEL_PATH}")
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

# 加载LoRA适配器
logger.info(f"开始加载LoRA适配器: {LORA_PATH}")
vl_model = PeftModel.from_pretrained(vl_model, LORA_PATH)

vl_model.eval()
min_pixels = 224 * 224
max_pixels = 448 * 448
vl_model_processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
)
logger.info("模型加载完成")

clear_memory()

def answer_with_mrag(text_query:str,
                     retrieval_top_k:int = 3,
                     reranker_top_k:int = 1,
                     max_new_tokens:int = 500,
                     question_idx:int = 0):
    # 初始化变量
    results = []
    results_rank = []
    
    # 简化模式：如果没有检索模型，直接使用第一个图像
    if retrieval_model is None:
        logger.warning("检索模型不可用，使用简化模式")
        # 使用第一个图像作为示例
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]
        if image_files:
            grouped_images_rank = [Image.open(os.path.join(IMAGE_PATH, image_files[0]))]
        else:
            logger.error("未找到图像文件")
            return "无法找到相关图像", results, results_rank
    else:
        try:
            # 检索阶段
            results = retrieval_model.search(text_query, k=retrieval_top_k)
            grouped_images = get_grouped_images(results, all_images)
            base64_list = images_to_base64(grouped_images)
            
            
            if reranker_model is None:
                logger.warning("重排序模型不可用，使用原始结果")
                grouped_images_rank = grouped_images[:reranker_top_k]
            else:
                results_rank = reranker_model.rank(text_query, base64_list)
                grouped_images_rank = process_ranker_results(results_rank, grouped_images, top_k=reranker_top_k)
        except Exception as e:
            logger.error(f"检索过程出错: {e}")
            # 回退到简化模式
            image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]
            if image_files:
                grouped_images_rank = [Image.open(os.path.join(IMAGE_PATH, image_files[0]))]
            else:
                logger.error("未找到图像文件")
                return "无法找到相关图像", results, results_rank

    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in grouped_images_rank]
            + [{"type": "text", "text": text_query}],
        }
    ]
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    return response,results,results_rank

# text_query = "介绍colpali"
# print(answer_with_mrag(text_query))



EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

# 只有在需要时才初始化DeepSeek模型
if DS_API_KEY is not None:
    eval_chat_model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=DS_API_KEY
    )
else:
    eval_chat_model = None
    logger.info("跳过DeepSeek模型初始化")

def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluation_prompt_template,
    field_prefix="deepseek-chat"
):
    answers = json.load(open(answer_path))
    for ex in tqdm(answers, desc="Evaluating"):
        if f"eval_score_{field_prefix}" in ex:
            continue 

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=ex["question"],
            response=ex["generated_answer"],
            reference_answer=ex["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        try:
            feedback, score = [s.strip() for s in eval_result.content.split("[RESULT]")]
            score = int(score)
        except Exception as e:
            feedback, score = f"ParseError: {e}\n{eval_result.content}", -1

        ex[f"eval_score_{field_prefix}"] = score
        ex[f"eval_feedback_{field_prefix}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
            
if __name__ == '__main__':
    logger.info("开始生成RAG答案...")
    logger.info("答案生成完成")
    
    # 跳过DeepSeek评估（因为API_KEY为None）
    if DS_API_KEY is not None and eval_chat_model is not None:
        logger.info("开始DeepSeek评估...")
        evaluate_answers(ANS_PATH,eval_chat_model,evaluation_prompt_template)
        result = pd.DataFrame(json.load(open(ANS_PATH, "r")))
        
        # 动态检测可用的评估字段
        eval_fields = [col for col in result.columns if col.startswith("eval_score_")]
        if eval_fields:
            eval_field = eval_fields[0]  # 使用第一个可用的评估字段
            logger.info(f"使用评估字段：{eval_field}")
            result[eval_field] = result[eval_field].apply(lambda x: float(x) if isinstance(x, int) else 1)
            result[eval_field] = (result[eval_field] - 1) / 4.0
            logger.info(f"评估平均分数: {result[eval_field].mean():.4f}")
            print(f"评估平均分数: {result[eval_field].mean():.4f}")
        else:
            logger.warning("没有找到评估结果字段")
        
        # 保存最终的DeepSeek评估结果
        final_timestamp = datetime.now()
        final_results = result.to_dict('records')
        
    else:
        logger.info("跳过DeepSeek评估（API_KEY为None或模型未初始化）")
        result = pd.DataFrame(json.load(open(ANS_PATH, "r")))
        logger.info(f"生成了 {len(result)} 条RAG答案")
        print(f"RAG评估完成，生成了 {len(result)} 条答案")
        
        # 保存最终结果（即使没有DeepSeek评估）
        final_timestamp = datetime.now()
        final_results = result.to_dict('records')
        
        # 显示一些示例结果
        if len(result) > 0:
            print("\n示例结果:")
            for i, row in result.head(3).iterrows():
                print(f"问题 {i+1}: {row['question'][:100]}...")
                print(f"生成答案: {row['generated_answer'][:100]}...")
                print("-" * 50)
    