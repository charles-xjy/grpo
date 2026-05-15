# GRPO 遥感变化检测数据构建与训练

本项目用于为遥感视觉语言模型构建 SFT 训练数据，并提供 GRPO 训练所需的自定义奖励函数。

## 项目结构

```
grpo/
├── construct_sft_data.py   # 构建变化检测 CoT-SFT 数据
├── construct_vqa_data.py   # 构建单图双尺度描述与 VQA 数据
└── plugin.py               # GRPO 训练的 LLM-as-Judge 奖励函数插件
```

## 数据集

支持以下三个遥感数据集，需放置于 `DATASET_ROOT`（默认 `/home/charles/mycode/sft+rl/dataset`）下：

| 数据集 | 场景类型 | 目录结构 |
|--------|----------|----------|
| **EBD** | 地质灾害（灾前/灾后对比） | `EBD/{event}/images/{id}_{pre/post}_disaster.*` |
| **LEVIR-CD+** | 城市建筑物变化检测 | `LEVIR-CD+/{train\|test}/{A\|B}/*.png` |
| **SECOND** | 多类地表覆盖变化检测 | `SECOND/{train\|test}/{im1\|im2}/*.png` |

## 数据构建流程

### 1. CoT-SFT 数据（变化检测）

`construct_sft_data.py` 使用本地部署的教师模型对三个数据集的图像对进行推理，生成专业分析文本，输出 SFT 格式的 JSONL 文件。

**采样策略：**
- EBD：每种灾害类型取前 200 对
- LEVIR-CD+：顺序取前 500 对
- SECOND：顺序取前 2000 对

**输出文件：**
```
EBD_sft.jsonl
LEVIR-CD+_sft.jsonl
SECOND_sft.jsonl
```

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "<image>\n<image>\n{随机指令}"},
    {"role": "assistant", "content": "{教师模型生成的分析报告}"}
  ],
  "images": ["/path/to/pre.png", "/path/to/post.png"]
}
```

支持**断点续传**：重新运行会自动跳过已处理的图像对。

### 2. 双尺度描述 + VQA 数据（单图）

`construct_vqa_data.py` 从未被 CoT 数据占用的剩余图像中，生成三类数据，输出到 `vqa_data/` 目录。

**流程（每张图像）：**
1. 调用教师模型生成**双尺度描述**（宏观场景 + 微观地物清单）
2. 基于描述，纯文本生成 4 个**开放式问答**（Q1）
3. 基于描述，纯文本生成 5 个**具体关系/计数问答**（Q2）

**采样上限：** 每个数据集最多取 3000 张图像（排除 CoT 已使用的部分）。

**输出文件：**
```
vqa_data/
├── dual_scale_desc.jsonl   # 单图双尺度解译
├── vqa_openended.jsonl     # 开放式问答
└── vqa_specific.jsonl      # 空间关系/计数问答
```

同样支持**断点续传**。

## 环境依赖

```bash
pip install openai tqdm
```

教师模型需在本地以 OpenAI 兼容接口方式部署（默认端口 `8001`），例如使用 swift：

```bash
swift deploy --model Qwen/Qwen3-VL-32B-Instruct --port 8001 --infer_backend vllm
```

## 运行数据构建脚本

```bash
# 构建变化检测 SFT 数据
python construct_sft_data.py

# 构建单图描述与 VQA 数据
python construct_vqa_data.py
```

## GRPO 奖励函数（plugin.py）

`plugin.py` 为 [ms-swift](https://github.com/modelscope/ms-swift) 框架提供自定义 GRPO 奖励插件，使用 LLM-as-Judge 对模型输出进行多维度打分。

### 评分维度

| 维度 | 满分 | 评估内容 |
|------|------|----------|
| 结构完备性 | 2.0 | 是否覆盖灾害判定、损害评估、防治建议三大模块 |
| 图像锚定精确度 | 3.0 | 分析是否锁定图像中的具体位置，拒绝通用模板 |
| 术语使用与地质逻辑 | 3.0 | 专业词汇准确性与物理因果逻辑 |
| 负面约束执行力 | 2.0 | 无开场白/结束语，不确定处使用"疑似"等词 |

总分范围：**0 ~ 10**，四个维度并行异步打分后求和。

### 配置方式

通过环境变量控制奖励模型连接：

```bash
export GENRM_API_BASE=http://localhost:8001/v1   # 默认值
export GENRM_TEMPERATURE=0.3                      # 默认值
```

### 在 swift GRPO 训练中使用

```bash
swift rlhf \
  --rlhf_type grpo \
  --external_plugins /path/to/plugin.py \
  --reward_funcs LLM_as_judger \
  ...
```
