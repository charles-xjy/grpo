# GRPO 遥感变化检测数据构建与训练

本项目用于为遥感视觉语言模型构建 SFT 训练数据，并提供 GRPO 训练所需的自定义奖励函数。

## 项目结构

```
grpo/
├── construct_sft_data.py   # 构建变化检测 CoT-SFT 数据
├── construct_vqa_data.py   # 构建单图双尺度描述与 VQA 数据
├── plugin.py               # GRPO 训练的 LLM-as-Judge 奖励函数插件
└── test_vllm.py            # 测试 vLLM 多模态接口的单次请求脚本
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

## vLLM 部署

教师模型需在本地以 OpenAI 兼容接口方式部署（默认端口 `8001`）：

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 2 \
  --port 8001 \
  --no-enable-chunked-prefill   # Qwen3-VL 必须加，否则多图并发会触发 deepstack buffer 溢出
```

或使用 swift：

```bash
swift deploy --model Qwen/Qwen3-VL-32B-Instruct --port 8001 --infer_backend vllm
```

## 并行加速机制

两个数据构建脚本均针对大规模数据生成进行了并行优化，是本项目的核心亮点。

### construct_sft_data.py — 多线程并行

基于 `ThreadPoolExecutor` + `as_completed`，所有图像对任务一次性提交入队，始终保持 `--concurrency` 个线程同时向 vLLM 发送请求。哪个请求先返回就先写入文件，不阻塞其他线程。

```
提交阶段：1000 个任务全部入队（瞬间完成）
          ↓
执行阶段：Thread-1 ──── 请求A ──── 写入
          Thread-2 ──── 请求B ──── 写入     ← 始终保持 N 个并发
          Thread-3 ──── 请求C ──── 写入
          Thread-4 ──── 请求D ──── 写入
                   请求D完成 → Thread-4 立即取下一个任务
```

文件写入通过 `threading.Lock` 保证线程安全。

### construct_vqa_data.py — 异步并行（每图 3 路并发）

基于 `asyncio` + `Semaphore`，每张图像内部也有并行：生成双尺度描述后，Q1（开放式问答）和 Q2（关系/计数问答）两路纯文本请求通过 `asyncio.gather` 同时发出，单张图像总耗时约等于"描述生成 + max(Q1, Q2)"而非三者之和。

```
每张图像的处理流程：

Step 1 │ 双尺度描述（含图，占 1 槽）
       ↓ 描述生成完成
Step 2 │ Q1 生成（纯文本）┐
       │ Q2 生成（纯文本）┘← asyncio.gather 并行，各占 1 槽
       ↓ 两路同时完成
       写入三个输出文件
```

全局并发上限由 `asyncio.Semaphore(concurrency)` 统一控制，所有图像的所有子任务共享这个槽位池。

## 运行数据构建脚本

```bash
# 构建变化检测 SFT 数据
python construct_sft_data.py [--base-url URL] [--concurrency N] [--retries N]

# 构建单图描述与 VQA 数据
python construct_vqa_data.py [--base-url URL] [--concurrency N] [--retries N]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动探测 | 指定模型名，也可通过 `$VLLM_MODEL` 环境变量设置 |
| `--base-url` | `http://localhost:8001` | vLLM 服务地址 |
| `--concurrency` | `4` | 并行请求数 |
| `--retries` | `3` | 单条记录失败后最大重试次数 |

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
