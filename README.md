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

支持**断点续传**：重新运行会自动跳过已处理的图像对（详见[断点续传机制](#断点续传机制)）。

### 2. 双尺度描述 + VQA 数据（单图）

`construct_vqa_data.py` 从未被 CoT 数据占用的剩余图像中，生成三类数据，输出到 `vqa_data/` 目录。

**图像采集策略：只取 post 侧图像**（EBD 取 post_disaster，LEVIR 取 B，SECOND 取 im2），避免 pre/post 同场景内容重复。CoT 占用集合同步只计 post 侧，共 3900 张。

**三阶段顺序执行：**

```
Phase 1  双尺度描述（含图推理，耗时最长）
         ↓ 全量完成后加载 desc_map
Phase 2  Q1 开放式问答（纯文本，依赖 Phase 1 的描述）
         ↓
Phase 3  Q2 具体关系/计数问答（纯文本，依赖 Phase 1 的描述）
```

每阶段各自独立断点续传，重启时自动跳过已完成部分，无需等待前序阶段重跑。

**采样上限：** 每个数据集最多取 3000 张 post 图像（排除 CoT 已使用的部分）。

**输出文件：**
```
vqa_data/
├── dual_scale_desc.jsonl   # 单图双尺度解译
├── vqa_openended.jsonl     # 开放式问答
└── vqa_specific.jsonl      # 空间关系/计数问答
```

## 环境依赖

```bash
pip install openai tqdm
```

## vLLM 部署

教师模型需在本地以 OpenAI 兼容接口方式部署（默认端口 `8001`）：

```bash
CUDA_VISIBLE_DEVICES=1,2 vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 2 --max-model-len 12000 \
  --enforce-eager \
  --gpu-memory-utilization 0.95 --port 8001
```

> **为什么必须加 `--enforce-eager`：** Qwen3-VL 的 deepstack 机制要求 buffer 按图像分辨率动态分配。vLLM 0.20.0 在 CUDA graph 模式下按 `total_num_scheduled_tokens` 静态分配 buffer，当 prefill 请求与 decode 请求被 batch queue 合并后，总 token 数可能小于单图所需的 deepstack tokens（例如 278 < 288），导致 EngineCore 崩溃。`--enforce-eager` 禁用 CUDA graph，改为动态分配，彻底规避此问题。`--no-enable-chunked-prefill` 对此无效。

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

### construct_vqa_data.py — 异步三阶段并行

基于 `asyncio` + `Semaphore`，三个阶段顺序执行，每阶段内部所有图像任务并行，由全局 `Semaphore(concurrency)` 限制同时打到 vLLM 的请求数。

```
Phase 1（所有图像并行）：
  image_1 → desc  ─┐
  image_2 → desc   ├─ Semaphore(4) 控制并发
  ...              │
  image_N → desc  ─┘
           ↓ 全部完成，加载 desc_map
Phase 2（所有图像并行）：
  (image_1, desc_1) → Q1 ─┐
  ...                      ├─ Semaphore(4)
Phase 3（所有图像并行）：
  (image_1, desc_1) → Q2 ─┐
  ...                      ├─ Semaphore(4)
```

进度条每个阶段独立显示，每完成一张图立即更新。

## 断点续传机制

两个数据构建脚本均支持断点续传，脚本崩溃、手动 Ctrl+C、vLLM 服务重启后，重新运行即可从断点继续，不会浪费已完成的计算。

### 核心原理

**将输出文件视为"已完成记录"的来源。** 每次启动时：

1. 读取已存在的输出 JSONL 文件，逐行解析
2. 提取每条记录中的 `images[0]`（图像路径）作为唯一标识，存入 set
3. 从完整待处理列表中过滤掉已完成的，只处理剩余部分
4. 以 **追加模式** (`"a"`) 打开输出文件，每处理完一条立即 `write + flush` 落盘

```
脚本启动
  │
  ├─ 1. 构建完整的待处理列表
  │
  ├─ 2. 读取输出 JSONL → 提取 images[0] → done_set
  │      ┌─ 文件存在 → done_set 有数据
  │      └─ 文件不存在 → done_set 为空
  │
  ├─ 3. remaining = 全量 - done_set
  │     打印: 总计 X | 已完成 Y | 待处理 Z
  │
  ├─ 4. 以 append 模式处理 remaining
  │     每成功一条 → 立即写入 + flush（保证落盘）
  │     失败 → 记入 fail_list，不写入文件
  │
  └─ 5. 打印统计，失败的记录下次运行自动重试
```

### SFT 脚本（单文件）

`construct_sft_data.py` 的断点粒度是**单条 pair**。每个输出 JSONL 文件独立判断，`images[0]` 对应 pre 图路径。

### VQA 脚本（三阶段各自独立断点续传）

`construct_vqa_data.py` 采用三阶段架构，每阶段独立检查自己的输出文件：

- **Phase 1**：检查 `dual_scale_desc.jsonl`，跳过已有描述的图像
- **Phase 2**：检查 `vqa_openended.jsonl`，跳过已有 Q1 的图像（需 Phase 1 已生成对应描述）
- **Phase 3**：检查 `vqa_specific.jsonl`，跳过已有 Q2 的图像（需 Phase 1 已生成对应描述）

运行时日志示例：

```
[Phase 1] 双尺度描述  已完成 1500 / 6147, 待处理 4647
Phase 1 双尺度描述: 100%|████████| 4647/4647 [1:02:13<00:00,  1.24it/s]

描述映射加载: 6147 条

[Phase 2] Q1 开放式问答  已完成 1480 / 6147, 待处理 4667
[Phase 3] Q2 具体关系/计数  已完成 1470 / 6147, 待处理 4677
```

### 失败恢复场景速查

| 场景 | 行为 |
|------|------|
| 脚本中途崩溃 / Ctrl+C | 已写入的记录已落盘，下次自动跳过 |
| 某条记录重试耗尽仍失败 | 不写入文件，下次运行重新尝试 |
| vLLM 服务重启 | 当前正在处理的请求触发重试（指数退避 2ⁿ 秒），重试耗尽则记入 fail_list |
| 输出文件被误删 | done_set 为空，该文件对应的所有记录从头生成 |
| 想强制重新处理某条记录 | 在输出 JSONL 中删除对应行，重新运行即可 |

### 注意事项

- **唯一标识是 `images[0]` 字符串**：因此不要移动图像文件目录，否则路径变化会导致断点续传失效（新旧路径不匹配，已完成的记录无法被识别）。
- **VQA 三阶段独立续传**：desc/Q1/Q2 各自独立判断完成状态。若某图 Phase 1 desc 已完成但 Phase 2 Q1 失败，下次重启只会在 Phase 2 重跑该图的 Q1，不影响 Phase 1 已有的描述。

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
export GENRM_CONCURRENCY=4                        # 全局并发上限，默认 4
```

**并发控制：** 单个 completion 的 4 个维度通过 `asyncio.gather` 同时打分；全局 `asyncio.Semaphore(GENRM_CONCURRENCY)` 限制同时打到 vLLM 的请求总数，避免多 completion 并发时压垮服务。

### 在 swift GRPO 训练中使用

```bash
swift rlhf \
  --rlhf_type grpo \
  --external_plugins /path/to/plugin.py \
  --reward_funcs LLM_as_judger \
  ...
```
