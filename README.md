# 📂 资源下载与配置指南 (Resources & Models)

本仓库的相关数据与预训练模型托管于 Hugging Face。请根据以下说明配置本地路径并下载相关权重。

---

## 📊 1. 数据集 (Datasets)
所有的训练、验证及微调数据请统一存放至本地 `data_download` 文件夹。

| 资源名称 | 类型 | 来源链接 | 建议本地路径 |
| :--- | :--- | :--- | :--- |
| **SFT RAG Dataset** | 核心数据集 | [🔗 点击访问](https://huggingface.co/datasets/Criy/sft_rag) | `data_download/` |

> **提示**：建议使用 `huggingface-cli` 批量下载数据：
> ```bash
> huggingface-cli download Criy/sft_rag --local-dir ./data_download --repo-type dataset
> ```

---

## 🤖 2. 预训练模型 (Models)
所有模型权重请统一存放在本地 `model_download` 文件夹中。本项目主要依赖 **Qwen2-VL** 系列及其变体模型。

### 📌 模型列表
| 模型系列 | 版本 | 官方托管链接 |
| :--- | :--- | :--- |
| **ColQwen2** | Base (基础版) | [🔗 vidore/colqwen2-base](https://huggingface.co/vidore/colqwen2-base) |
| **ColQwen2** | v1.0 | [🔗 vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) |
| **MonoQwen2** | VL-v0.1 | [🔗 lightonai/MonoQwen2-VL-v0.1](https://huggingface.co/lightonai/MonoQwen2-VL-v0.1) |
| **Qwen2-VL** | 2B-Instruct | [🔗 Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| **Qwen2.5-VL** | 7B-Instruct | [🔗 Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

---

## ⚙️ 3. 快速初始化指令
在开始运行项目前，请确保在根目录下创建了对应的存储目录：

```bash
# 创建数据与模型下载目录
mkdir -p data_download
mkdir -p model_download
