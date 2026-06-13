# 带分析过程的遥感 VQA SFT 数据构建

本项目当前聚焦于构建**单图遥感 VQA SFT 冷启动数据**。每条样本不仅包含最终答案，还包含可监督的分析过程，用于训练模型按照“定位目标、观察证据、给出判断”的方式回答遥感图像问题。

核心目标不是生成长篇遥感解译报告，而是为后续 GRPO 提供一个足够稳定的遥感 VQA 初始模型：模型先学会看遥感图、理解问题、按固定格式回答，再通过更高质量的 GRPO 数据继续提升识别准确率。

## 项目结构

```text
grpo/
├── construct_vqa_data.py      # 构建单图描述、问题和带分析过程的 VQA SFT 数据
├── augment_vqa_prompts.py     # 对用户问题做表达多样化增强
├── construct_sft_data.py      # 历史脚本：双时相变化检测 SFT 数据构建
├── plugin.py                  # 历史脚本：GRPO 训练奖励函数插件
└── test_vllm.py               # 测试 vLLM 多模态接口
```

当前推荐主线只使用 `construct_vqa_data.py`。`construct_sft_data.py` 和 `plugin.py` 保留为历史能力，不作为本阶段 SFT 数据构建重点。

## 训练分工

推荐的整体训练路线：

```text
SFT:
  使用本项目构建的带 analysis 的遥感 VQA 数据
  目标：学会遥感 VQA 格式、基础识别、证据化回答

GRPO:
  使用 GeoChat_Instruct 等更高质量遥感 instruction/reference 数据
  目标：进一步提升识别准确率和答案可靠性
```

SFT 阶段训练的是“会答、会按格式答、会用遥感证据答”。GRPO 阶段再重点优化“答得更准”。

因此，SFT 数据可以容忍少量噪声，但不能形成系统性坏习惯。需要重点避免：

- 答案过长，变成长篇遥感报告。
- `<analysis>` 编造图像中不可见的证据。
- 遇到不确定目标仍强行判断。
- 单图问题回答行政建议、灾害成因或工程修复方案。
- 计数题编造过精确数字。
- 问题和答案不匹配。

推荐的 SFT 数据形态：

```text
短问题
+ 可见证据分析
+ 简洁答案
+ 自动过滤
+ 抽样人工审核
```

## 数据集

支持以下三个遥感数据集，需放置于 `DATASET_ROOT` 下：

| 数据集 | 使用图像 | 场景类型 | 目录结构 |
|--------|----------|----------|----------|
| **EBD** | `post_disaster` | 灾后遥感场景 | `EBD/{EVENT}/images/{EVENT}_{ID}_post_disaster.*` |
| **LEVIR-CD+** | `B` | 城市建筑与道路场景 | `LEVIR-CD+/{train\|test}/B/*.png` |
| **SECOND** | `im2` | 多类地表覆盖场景 | `SECOND/{train\|test}/im2/*.png` |

VQA 阶段只使用单张图像。对于双时相数据集，只取后时相或 post 侧图像，避免同一场景的 pre/post 内容重复进入单图 VQA。

## 数据目标

重构后的 SFT 数据训练目标是：

- 会识别遥感图像中的基础地物。
- 会理解短 VQA 问题。
- 会用可见证据组织简短分析。
- 会输出简洁答案。
- 遇到不确定内容时会说明不确定，而不是强行编造。

最终样本采用 OpenAI/Swift 常见的多模态 SFT JSONL 格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\n图像右侧道路附近是否分布有建筑？"
    },
    {
      "role": "assistant",
      "content": "<analysis>\n1. 定位：问题关注图像右侧道路附近区域。\n2. 观察：该区域可见浅灰色线状道路，道路两侧有多个规则矩形屋顶。\n3. 判断：规则几何形态和沿道路分布特征符合建筑分布特征。\n</analysis>\n<answer>是，图像右侧道路附近分布有多个建筑。</answer>"
    }
  ],
  "images": ["/path/to/image.png"],
  "meta": {
    "task_type": "vqa_with_analysis",
    "question_type": "existence",
    "source_dataset": "LEVIR-CD+"
  }
}
```

`assistant` 必须同时包含：

- `<analysis>`：结构化证据分析，通常 2-4 步。
- `<answer>`：最终答案，简洁直接。

不推荐使用 `<think>` 作为主训练字段。本项目需要的是可监督的“证据分析过程”，不是开放式内部思维链。

## 构建流程

SFT 数据建议分为三个生成阶段、一个自动质检阶段和一个抽样人工质检阶段。

### Phase 1: 受控图像描述

教师模型先对单张遥感图像生成受控描述，作为后续问题和答案生成的证据参考。

描述应覆盖：

- 场景类型：城市、乡村、农田、水体、植被、工业区、灾后区域等。
- 主要地物：建筑、道路、植被、水体、车辆、裸地、人工设施等。
- 空间关系：相邻、位于某方向、沿道路分布、被道路分隔等。
- 可计数对象：只统计清晰可见的对象，不强行计入遮挡或模糊目标。
- 不确定内容：无法确认的用途、类型或数量需要明确标记。

描述阶段只允许写图像中可见的内容，不输出行政建议、灾害成因、工程治理方案、精确坐标或精确面积。

### Phase 2: 问题生成

基于 `图像 + 受控描述` 生成问题。问题应覆盖多种 VQA 能力，而不是只做开放式描述。

推荐问题类型：

| 类型 | 说明 | 建议占比 |
|------|------|----------|
| `existence` | 是否存在某类地物 | 15% |
| `attribute` | 颜色、形状、屋顶类型、材质或纹理 | 15% |
| `location` | 某地物位于图像哪个区域 | 15% |
| `spatial_relation` | 两个地物之间的相对位置或邻接关系 | 20% |
| `counting` | 清晰可见目标的数量或估计数量 | 10% |
| `scene` | 整体场景类型或主要功能区 | 10% |
| `comparison` | 不同区域的建筑、植被、水体或道路密度对比 | 10% |
| `reasoning` | 基于可见证据的简单判断 | 5% |

每张图建议生成 6-10 个问题。图像内容简单时应减少问题数量，不为了凑数生成低质量问答。

### Phase 3: 带分析过程的答案生成

基于 `图像 + 受控描述 + 问题 + question_type` 生成答案。输出格式固定为：

```text
<analysis>
1. 定位：...
2. 观察：...
3. 判断：...
</analysis>
<answer>...</answer>
```

不同问题类型使用不同分析模板：

| 问题类型 | 分析过程要求 |
|----------|--------------|
| `existence` | 定位目标类别或区域，观察是否存在符合形态、颜色、纹理的目标，再给出存在性判断。 |
| `attribute` | 定位目标地物，观察颜色、形状、纹理或屋顶特征，再给出属性。 |
| `location` | 识别目标地物，判断其在整幅图中的相对方位，再用左上、右下、中央等区域描述。 |
| `spatial_relation` | 分别定位两个目标，比较二者的方位、邻接、间隔关系，再给出空间关系。 |
| `counting` | 确定计数对象，统计清晰可见目标；若不确定，答案中说明为估计值。 |
| `scene` | 概括主要地物组成，比较建筑、道路、植被、水体等占比，再判断场景类型。 |
| `comparison` | 分别观察两个区域或对象，比较密度、数量、面积或显著程度，再给出结论。 |
| `reasoning` | 只基于可见证据做短链路判断，不扩展到图像外知识。 |

### Phase 4: 自动质检与分级

生成后需要进行自动质检，再决定是否进入训练集。

最少检查：

- 是否是合法 JSONL。
- `messages` 和 `images` 字段是否完整。
- 图像路径是否存在。
- `assistant` 是否同时包含 `<analysis>` 和 `<answer>`。
- `<analysis>` 是否为 2-4 步。
- `<answer>` 是否简洁且回答了问题。
- `counting` 问题是否包含数量或估计数量。
- `location` 和 `spatial_relation` 问题是否包含方位或关系表达。
- 是否出现高风险幻觉词。

高风险内容包括：

- 精确坐标
- 精确面积
- 米级位移
- 灾害成因
- 行政处置
- 补偿建议
- GNSS
- InSAR
- 禁建区
- 工程修复方案

这些内容通常不适合作为单图 VQA 答案。出现后应降级或拒收。

### Phase 5: 抽样人工质检

自动质检后，需要对候选 SFT 数据做分层抽样人工审核。人工质检不做全量，目标是估计每类数据的真实质量，并发现系统性错误。

推荐抽样维度：

- `source_dataset`：EBD / LEVIR-CD+ / SECOND。
- `question_type`：existence / attribute / location / spatial_relation / counting / scene / comparison / reasoning。
- `auto_validation_label`：pass / warning。

推荐抽样规模：

```text
每个 dataset × 每个 question_type 抽 20 条
```

如果某个桶样本较少，可以使用：

```text
min(20, 桶内样本数的 10%)
```

人工审核字段建议：

```text
sample_id
source_dataset
question_type
image_path
question
analysis
answer
auto_label
image_supported
question_valid
analysis_grounded
answer_correct
hallucination
severity
final_label
comment
```

人工判定标准：

- `gold`：图像支持问题，analysis 基于可见证据，answer 正确简洁，无明显幻觉。
- `silver`：基本正确，存在轻微啰嗦或不精确，但不影响训练目标。
- `rejected`：问题不可答、答案错误、analysis 编造证据、出现图像外推断或计数明显错误。

抽检结果应反向影响数据发布。如果某个 `dataset/question_type` 桶的人工通过率过低，该桶全量样本不应直接进入 SFT 主训练集，需要回退重生成、修改 prompt 或降低采样比例。

## 推荐输出目录

```text
sft_data/
├── raw/
│   ├── structured_desc.jsonl
│   ├── questions.jsonl
│   └── answers_raw.jsonl
├── processed/
│   ├── sft_vqa_with_analysis_silver.jsonl
│   └── sft_vqa_with_analysis_gold.jsonl
├── audit/
│   ├── audit_sample.jsonl
│   ├── audit_sheet.csv
│   └── human_reviewed.csv
├── rejected/
│   ├── format_fail.jsonl
│   ├── hallucination_risk.jsonl
│   └── unanswerable.jsonl
└── reports/
    ├── sft_stats.json
    └── audit_report.json
```

推荐分级：

- `gold`：人工审核通过或高置信规则通过的数据，可用于加权训练或 sanity check。
- `silver`：自动规则校验通过，且所在抽检桶质量达标的主训练数据。
- `rejected`：格式错误、明显幻觉、重复或无训练价值的数据。

## 采样策略

不要直接按文件顺序取前 N 张。推荐使用固定随机种子的分层采样。

分层维度：

- 数据集：EBD / LEVIR-CD+ / SECOND。
- 场景类型：城市、乡村、农田、水体、植被、工业区、灾后场景。
- 复杂度：简单 / 中等 / 复杂。
- 地物密度：低 / 中 / 高。

每次采样应记录：

- `seed`
- 原始候选池数量
- 各数据集采样数量
- 样本图像路径
- 采样时间

这样后续重跑和对比实验才能复现。

## Prompt 约束

生成描述、问题和答案时，应始终保留以下约束：

```text
只允许依据图像中可见内容回答。
不要推断图像外信息。
不要给出行政、工程、灾害治理建议。
不要编造精确坐标、面积、距离或数量。
如果目标不清晰，必须说明“不确定”或“无法确认”。
分析过程必须是可见证据链，不要写开放式内心思考。
```

推荐答案风格：

```text
<analysis>
1. 定位：问题询问图像中央偏右的道路区域。
2. 观察：该区域可见浅灰色线状道路，道路两侧分布有多个规则矩形建筑屋顶。
3. 判断：道路与建筑紧邻，说明该区域属于道路连接的建筑分布区。
</analysis>
<answer>图像中央偏右的道路两侧分布有多个建筑。</answer>
```

不推荐答案风格：

```text
该区域可能是城市扩张导致的居民区建设，建议加强规划管理并开展后续监测。
```

这类内容超出了 VQA 的图像证据范围，容易污染训练目标。

## 环境依赖

```bash
pip install openai tqdm
```

## vLLM 部署

教师模型需要以 OpenAI 兼容接口方式部署。默认服务地址为 `http://localhost:8001`。

```bash
CUDA_VISIBLE_DEVICES=1,2 vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 2 --max-model-len 12000 \
  --enforce-eager \
  --gpu-memory-utilization 0.95 --port 8001
```

或使用 `swift`：

```bash
swift deploy --model Qwen/Qwen3-VL-32B-Instruct --port 8001 --infer_backend vllm
```

`--enforce-eager` 用于规避部分多模态模型在 vLLM CUDA graph 模式下的动态图像 token buffer 问题。

## 运行方式

```bash
python construct_vqa_data.py --base-url http://localhost:8001 --concurrency 4 --retries 3
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动探测 | 指定模型名，也可通过 `$VLLM_MODEL` 环境变量设置 |
| `--base-url` | `http://localhost:8001` | vLLM 服务地址 |
| `--concurrency` | `4` | 并发请求数 |
| `--retries` | `3` | 单条记录失败后的最大重试次数 |

## 断点续传

构建脚本应将输出 JSONL 视为完成状态来源：

1. 启动时读取已有输出文件。
2. 提取每条记录的 `images[0]` 作为唯一标识。
3. 从待处理图像列表中过滤已完成样本。
4. 每生成成功一条就立即追加写入并 `flush`。
5. 失败样本不写入主输出，下次运行自动重试。

注意：断点续传依赖图像路径字符串。移动数据集目录会导致旧路径无法匹配，已完成样本可能被重新生成。

## 数据质量原则

带分析过程的 VQA SFT 数据应遵循以下原则：

- 问题短而明确。
- 分析过程只写可见证据。
- 答案直接回答问题。
- 不把遥感报告当作 VQA 答案。
- 不把描述阶段的幻觉继续传递到问答阶段。
- 不为了数量牺牲问题质量。

本项目推荐的 SFT 冷启动目标是：模型能够先定位问题关注对象，再观察图像证据，最后给出简洁答案；后续再通过 GeoChat_Instruct 等数据进行 GRPO，提高遥感图像识别准确率。
