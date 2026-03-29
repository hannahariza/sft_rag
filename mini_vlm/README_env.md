# mini_vlm 运行环境说明

本文档描述本目录下 **Python 虚拟环境（`venv/`）的实际安装状态**，便于在其他机器上对照复现或排查依赖问题。

> **说明**：`requirements.txt` 仅列出项目**直接依赖**的钉版本；下列 **`pip freeze`** 输出为当前 venv 的**完整已安装包列表**（含传递依赖），与 `pip` 解析结果一致。生成时 `pip check` 无依赖冲突。

---

## 1. 环境概览

| 项目 | 值 |
|------|-----|
| 解释器 | Python **3.12.3** |
| 包管理 | **pip 25.2** |
| 虚拟环境路径（相对本目录） | `./venv` |
| 是否使用系统 site-packages | **否**（`include-system-site-packages = false`，隔离环境） |
| 深度学习栈 | **PyTorch 2.5.1 + CUDA 11.8**（`torch` / `torchvision` / `torchaudio` 带 `+cu118` 后缀） |

`venv/pyvenv.cfg` 中记录的创建命令指向历史路径；**请以当前仓库中的 `venv` 实际位置为准**（本仓库：`sft_rag/mini_vlm/venv`）。

---

## 2. 激活环境

在 `mini_vlm` 目录下：

```bash
source venv/bin/activate   # Linux / macOS
# Windows: venv\Scripts\activate
```

验证：

```bash
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 与 `requirements.txt` 的关系

- **`requirements.txt`**：项目维护的**直接依赖**列表（其中 `torch` 在文件中注释为「单独安装」）。
- **本文档中的 `pip freeze`**：反映 **venv 当前真实状态**，包含所有传递依赖（如 `langchain-core`、`accelerate`、`nvidia-*` CUDA 运行时轮子等）。

若需在新环境复现「接近当前机器」的完整环境，可将下文 **§5** 中内容保存为文件后执行：

```bash
python -m pip install -r requirements.txt
# 再按 PyTorch 官方说明安装 torch 2.5.1+cu118（或使用当时保存的完整 freeze）
```

单独安装带 CUDA 的 PyTorch 时，通常需使用 PyTorch 提供的 **CUDA 11.8** 索引，例如（版本以官方文档为准）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. 主要组件（节选）

以下为与 mini_vlm / VLM-RAG 相关、且常在 `requirements.txt` 中出现的核心包及其**当前 venv 中的版本**（与 `pip freeze` 一致）：

| 组件 | 版本 |
|------|------|
| torch | 2.5.1+cu118 |
| torchvision | 0.20.1+cu118 |
| torchaudio | 2.5.1+cu118 |
| transformers | 4.51.3 |
| trl | 0.11.4 |
| peft | 0.14.0 |
| deepspeed | 0.15.4 |
| datasets | 3.2.0 |
| colpali_engine | 0.3.10 |
| Byaldi | 0.0.7 |
| qwen-vl-utils | 0.0.11 |
| langchain | 0.3.25 |
| langchain-deepseek | 0.1.3 |
| huggingface-hub | 0.30.2 |

完整列表见下一节。

---

## 5. 完整依赖快照（`pip freeze`）

以下为在 **`venv` 激活等价路径** 下执行 `python -m pip freeze` 得到的输出，用于锁定**实际**环境：

```
accelerate==1.10.1
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.10.0
asttokens==3.0.0
attrs==25.3.0
av==15.1.0
backcall==0.2.0
Byaldi==0.0.7
catalogue==2.0.10
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.2.1
colpali_engine==0.3.10
datasets==3.2.0
decorator==5.2.1
deepspeed==0.15.4
dill==0.3.8
distro==1.9.0
docstring_parser==0.17.0
eval_type_backport==0.2.2
executing==2.2.1
filelock==3.19.1
frozenlist==1.7.0
fsspec==2024.9.0
gitdb==4.0.12
GitPython==3.1.45
greenlet==3.2.4
h11==0.16.0
hf-xet==1.1.10
hjson==3.1.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.30.2
idna==3.10
ipython==8.12.3
jedi==0.19.2
Jinja2==3.1.6
jiter==0.10.0
joblib==1.5.2
jsonlines==4.0.0
jsonpatch==1.33
jsonpointer==3.0.0
langchain==0.3.25
langchain-core==0.3.76
langchain-deepseek==0.1.3
langchain-openai==0.3.33
langchain-text-splitters==0.3.11
langsmith==0.3.45
markdown-it-py==4.0.0
MarkupSafe==3.0.2
matplotlib-inline==0.1.7
mdurl==0.1.2
ml_dtypes==0.5.3
modelscope==1.29.2
mpmath==1.3.0
msgpack==1.1.1
mteb==1.6.35
multidict==6.6.4
multiprocess==0.70.16
networkx==3.5
ninja==1.13.0
numpy==2.3.3
nvidia-cublas-cu11==11.11.3.6
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu11==9.1.0.70
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu11==10.9.0.58
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu11==10.3.0.86
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu11==11.7.5.86
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu11==2.21.5
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu11==11.8.86
nvidia-nvtx-cu12==12.4.127
openai==1.107.2
orjson==3.11.3
packaging==25.0
pandas==2.2.3
parso==0.8.5
pdf2image==1.17.0
peft==0.14.0
pexpect==4.9.0
pickleshare==0.7.5
pillow==11.2.1
platformdirs==4.4.0
prompt_toolkit==3.0.52
propcache==0.3.2
protobuf==6.32.1
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
py-cpuinfo==9.0.0
pyarrow==21.0.0
pydantic==2.11.8
pydantic_core==2.33.2
Pygments==2.19.2
python-dateutil==2.9.0.post0
pytrec_eval-terrier==0.5.9
pytz==2025.2
PyYAML==6.0.2
qwen-vl-utils==0.0.11
regex==2025.9.1
requests==2.32.5
requests-toolbelt==1.0.0
rerankers==0.9.1.post1
rich==14.1.0
safetensors==0.5.3
scikit-learn==1.7.2
scipy==1.16.2
sentence-transformers==5.1.0
sentry-sdk==2.37.1
setuptools==80.9.0
shtab==1.7.2
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
SQLAlchemy==2.0.43
srsly==2.5.1
stack-data==0.6.3
sympy==1.13.1
tenacity==9.1.2
threadpoolctl==3.6.0
tiktoken==0.11.0
tokenizers==0.21.4
torch==2.5.1+cu118
torchaudio==2.5.1+cu118
torchvision==0.20.1+cu118
tqdm==4.67.0
traitlets==5.14.3
transformers==4.51.3
triton==3.1.0
trl==0.11.4
typeguard==4.4.4
typing-inspection==0.4.1
typing_extensions==4.15.0
tyro==0.9.31
tzdata==2025.2
urllib3==2.5.0
wandb==0.21.4
wcwidth==0.2.13
xxhash==3.5.0
yarl==1.20.1
zstandard==0.23.0
```

---

## 6. 依赖自检

在本文档编写时，于该 venv 中执行：

```bash
python -m pip check
```

输出为：**`No broken requirements found.`**

---

## 7. 维护建议

- 升级或增删包后，可重新执行 `pip freeze` 更新 **§5**，并注明日期。
- 若仅需最小可运行集，以 **`requirements.txt`** 为准；若需逐包对齐当前机器，以 **§5** 为准。
