# GRPO 遥感变化检测数据构建与训练

本项目用于为遥感视觉语言模型构建 SFT 训练数据，并提供 GRPO 训练所需的自定义奖励函数。

## 项目结构

```
grpo/
├── data/                          # LLaMA-Factory 数据集注册
│   └── dataset_info.json
├── sft_data/                      # Stage 2 SFT 数据（图像对）
│   ├── EBD_sft.jsonl              #   地震灾害对比 (1400条)
│   ├── LEVIR-CD+_sft.jsonl        #   建筑物变化检测 (500条)
│   └── SECOND_sft.jsonl           #   地表覆盖变化 (2000条)
├── vqa_data/                      # Stage 1 VQA 数据（单图）
│   ├── dual_scale_desc.jsonl      #   双尺度描述
│   ├── vqa_openended.jsonl        #   开放式问答
│   └── vqa_specific.jsonl         #   空间关系/计数问答
│
├── construct_sft_data.py          # 数据构造：教师模型生成 SFT 数据
├── construct_vqa_data.py          # 数据构造：教师模型生成 VQA 数据
├── construct_grpo_data.py         # 数据构造：生成仅含提问的 GRPO 数据
├── score_evaluation.py            # 测评打分：教师模型作为裁判
├── evaluate_model.py              # 模型测评：教师出题，学生答题
├── augment_vqa_prompts.py         # 问题多样化改写
├── print_samples.py               # 打印数据文件前 N 条
│
├── sft_loar.yaml                  # Stage 1 训练配置
├── cot_loar.yaml                  # Stage 2 训练配置
│
├── sft_output/                    # Stage 1 LoRA 输出
├── sft_stage1_merged/             # Stage 1 合并后模型
│
├── plugin.py                      # GRPO 训练的 LLM-as-Judge 奖励函数
└── test_vllm.py                   # 测试 vLLM 多模态接口
```

## 数据集

支持以下三个遥感数据集，需放置于 `DATASET_ROOT`（默认 `/home/charles/mycode/sft+rl/dataset`）下：

| 数据集 | 场景类型 | 目录结构 |
|--------|----------|----------|
| **EBD** | 地质灾害（灾前/灾后对比） | `EBD/{EVENT}/images/{EVENT}_{ID}_{pre\|post}_disaster.*` |
| **LEVIR-CD+** | 城市建筑物变化检测 | `LEVIR-CD+/{train\|test}/{A\|B}/*.png` |
| **SECOND** | 多类地表覆盖变化检测 | `SECOND/{train\|test}/{im1\|im2}/*.png` |

## 数据构建流程

### 1. CoT-SFT 数据（变化检测）

`construct_sft_data.py` 使用本地部署的教师模型对三个数据集的图像对进行推理，生成专业分析文本，输出 SFT 格式的 JSONL 文件。

**采样策略：**
- EBD：每种灾害类型取前 20 对（共 7 事件，~140 对）
- LEVIR-CD+：顺序取前 100 对
- SECOND：顺序取前 100 对

### 2. 双尺度描述 + VQA 数据（单图）

`construct_vqa_data.py` 从未被 CoT 数据占用的剩余图像中，生成三类数据，输出到 `vqa_data/` 目录。每张图产出 1 + 3 + 3 = 7 条 Q&A 记录，三数据集共 ~2380 条。

### 3. GRPO Prompt 数据（强化学习）

`construct_grpo_data.py` 从剩余未使用的影像中采样，生成仅包含提问（Prompts）的数据集，用于后续的 GRPO 强化学习训练。

**采样规模：**
- EBD: 300 条
- LEVIR-CD+: 500 条
- SECOND: 1500 条
- **总计: 2300 条**

**数据格式：**
符合 MS-Swift 的多模态 `chat_dataset` 规范，且全部使用绝对路径以保证训练稳定性。
```json
{
  "messages": [{"role": "user", "content": "<image>\n<image>\n{随机人设+任务指令}"}],
  "images": ["/abs/path/pre.png", "/abs/path/post.png"]
}
```

## 环境依赖

```bash
pip install openai tqdm pandas pyarrow
```

## 训练流程

### Stage 1：VQA 单图训练
使用三个 VQA 数据集，在 Qwen3-VL-4B-Instruct 上训练第一轮 LoRA。
```bash
llamafactory-cli train sft_loar.yaml
```

### Stage 2：SFT 图像对训练
基于合并后的 Stage 1 模型，使用三个图像对数据集进行第二轮 SFT。
```bash
llamafactory-cli train cot_loar.yaml
```

### Stage 3：GRPO 强化学习训练

在完成两阶段 SFT 后，使用 GRPO (Group Relative Policy Optimization) 进一步优化模型的解译质量。

#### 1. 注册数据集
在 `data/dataset_info.json` 中添加：
```json
"remote_sensing_grpo":{
  "file_name": "grpo_swift.jsonl",
  "columns": {
    "prompt": "messages",
    "images": "images"
  }
}
```

#### 2. 启动训练

以下是针对 **3 张 5880 显卡 (单卡约 48GB 显存)** 且裁判模型（教师模型）为 **32B 级别大模型** 所深度优化的 GRPO 训练脚本。该脚本采用了 `colocate` 模式以最大化吞吐量，并对显存分配进行了极限切分。

```bash
#!/bin/bash

# 1. 显存碎片管理
# 允许 PyTorch 动态扩展已分配的显存段，极大降低跑大模型时长文本引发的显存碎片化 OOM 风险。
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 2. 奖励模型连接配置
# 裁判模型地址：指向本地 8002 端口的 32B 教师模型
export GENRM_API_BASE=http://localhost:8002/v1
# 裁判打分时的温度参数，0.3 偏向确定性与客观性
export GENRM_TEMPERATURE=0.3
# 控制打分插件并发打到 8002 端口的请求数，避免瞬间压垮 32B 模型
export GENRM_CONCURRENCY=1

# 3. 启动 Swift GRPO (严格基于 3 张 5880 显卡)
# 将 0, 1, 2 号共 3 张显卡全部投入训练进程
CUDA_VISIBLE_DEVICES=0 \
# 告诉 DeepSpeed 等分布式框架，当前节点启用了 3 个进程

# 强制限制输入图像最大像素约为 800x800，极大削减视觉 Token 数量，是防 OOM 的保命参数
MAX_PIXELS=602112 \
swift rlhf \
     --rlhf_type grpo \
     --model_type qwen3-vl-instruct \
     # 挂载 Stage 2 阶段 SFT 训练并合并后的模型权重
     --model_id_or_path /home/charles/mycode/grpo/model/sft_all_merged \
     # 加载我们在 dataset_info.json 中注册好的 GRPO 纯问题数据集
     --dataset remote_sensing_grpo \
     --template qwen3_vl \
     # 加载外部奖励计算插件（包含打分规则与网络请求逻辑）
     --external_plugins /home/charles/mycode/grpo/plugin.py \
     --reward_funcs LLM_as_judger \
     # 开启 vLLM 加速生成阶段（Rollout），显著提升强化学习速度
     --use_vllm true \
     # 核心优化：同驻模式。vLLM 推理引擎与 PyTorch 训练引擎共享显卡，省去跨进程通信开销
     --vllm_mode colocate \
     # 极限显存切分：vLLM 仅占用单卡 20% 显存（约 9.6G）
     --vllm_gpu_memory_utilization 0.2 \
     --vllm_tensor_parallel_size 1 \
     # 限制生成时的上下文最大长度
     --vllm_max_model_len 16384 \
     --torch_dtype bfloat16 \
     --num_train_epochs 1 \
     # 单卡 batch size 设为 1，多模态训练显存消耗极大，不可调大
     --per_device_train_batch_size 1 \
     # 梯度累加 16 次，相当于总 Batch Size = 3(卡) x 1 x 16 = 48
     --gradient_accumulation_steps 16 \
     --learning_rate 1e-6 \
     # 完美负载均衡：每针对一个 prompt 生成 3 个回答。因有 3 张卡，每张卡刚好生成 1 个，毫无闲置
     --num_generations 8 \
     # 生成回答的温度，0.9 能带来较高的随机性，利于 GRPO 进行优劣对比学习
     --temperature 0.9 \
     # 使用 Zero2 策略拆分优化器状态与梯度，节省显存
     --deepspeed zero2 \
     # 因 5880 显存较大，默认关闭 CPU 卸载以换取极限训练速度。如仍 OOM，请改为 true
     --offload_model false \
     --offload_optimizer false \
     # GRPO 训练数学参数：进行 Token 级别的重要度采样与优势值估计，使训练更加稳定
     --importance_sampling_level token \
     --advantage_estimator grpo \
     --epsilon 0.2 \
     # KL 惩罚系数，0.001 确保模型在学习新奖励的同时，不至于完全忘记 SFT 阶段的基础能力
     --beta 0.001 \
     # 模型一次最多能生成的文字 Token 长度
     --max_completion_length 4096 \
     --save_steps 100 \
     --logging_steps 1 \
     --report_to swanlab \
     --swanlab_project swift-grpo \
     --output_dir output_grpo
```

> **硬件避坑指南**：若同一台机器上同时驻留了 32B 教师模型与这个 3 卡训练进程，即便拥有 3×48GB 显存，仍极度容易在生成长文本时引发 OOM。**一旦发生显存溢出，请立即将 `--offload_optimizer false` 改为 `true`**。

---

## 模型评估体系 (Evaluation Methodology)

本项目建立了一套严谨的 **Teacher-as-a-Judge** 自动测评流水线，通过模拟“闭卷考试”来量化模型在遥感解译任务上的真实能力提升。

### 1. 测评维度设计

| 维度 | 任务描述 | 考察能力 |
| :--- | :--- | :--- |
| **Change Det (CoT)** | 针对前后两时相影像生成对比分析报告 | 变化敏感度、地学逻辑、专业表达 |
| **VQA - Open (Q1)** | 对单图场景进行宏观描述与开放式回答 | 基础视觉感知、自然语言对齐 |
| **VQA - Reason (Q2)** | 针对微小地物进行计数、方位及逻辑推理 | 精确目标定位、步进式推理 (Thinking) |

### 2. 测评工作流 (Pipeline)

测评过程分为三个阶段：**出题、考试、判卷**。

1.  **出题**：教师模型 (8002) 读取图像，生成双尺度描述，并基于描述出题（Q1/Q2）同时提供标准参考答案。
2.  **考试**：待测模型（学生或基线）仅凭图像和题目进行回答，不接触参考答案和描述。
3.  **判卷**：教师模型 (8002) 重新看图，评估学生回答的准确性与专业性。

### 3. 消融实验：对比基准 (Baseline)

为了验证微调效果，建议按以下顺序运行完整的对比流程：

```bash
# 1. 微调后模型测评 (学生 8001)
python3 evaluate_model.py

# 2. 微调前模型测评 (基准 8003)
python3 evaluate_baseline.py

# 3. 自动判卷 (教师 8002)
python3 score_evaluation.py                    # 判微调后
python3 score_evaluation.py --input-dir evaluation/baseline  # 判基准

# 4. 生成综合演进报告
python3 print_final_report.py
```

### 4. 关键特性

-   **全自动对比**：自动计算“提升幅度”百分比，直观展示训练收益。
-   **天花板基准**：评分脚本会让教师模型也参加“闭卷考试”，建立起该任务下的理论最高分（Skyline），避免由于题目过难导致的低分误判。
-   **断点续传**：所有测评与评分脚本均支持断点续传，遇到 API 报错重启即可，不浪费算力。

---

## 模型测评结果


本项目使用 **Teacher-as-a-Judge** 模式对训练后的模型进行闭卷测评。由教师模型（Port 8002）出题和判卷。

| 测评维度 | 学生得分 | 老师得分 | 达到老师水平 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **1. Change Detection (CoT)** | **9.46** | **9.88** | 95.7% | 评估报告的专业性、准确性与逻辑性 |
| **2. VQA - Open-ended (Q1)** | **7.63** | **9.32** | 81.9% | 评估对单图场景的宏观理解与描述 |
| **3. VQA - Reasoning (Q2)** | **4.14** | **4.96** | 83.5% | 评估对细节地物计数、方位及逻辑推理 |

**测评深度分析：**
- **CoT (9.46 vs 9.88)**：学生在变化检测报告的格式规范上已极其接近老师水平。
- **Q2 (4.14 vs 4.96)**：虽然分数较低，但老师模型在闭卷答题时也难以完全覆盖细节，学生已达到老师 **83.5%** 的性能。
