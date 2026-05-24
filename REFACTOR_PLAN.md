# 数据构造脚本重构方案（讨论稿）

## 一、现状分析

当前代码结构：

```
construct_sft_data.py      # 700行，同步 ThreadPoolExecutor，处理图像对
construct_vqa_data.py      # 700行，异步 asyncio，处理单图三阶段
augment_vqa_prompts.py     # 500行，离线改写问题文本
```

### 1.1 重复代码（两脚本间）

| 重复项 | construct_sft_data.py | construct_vqa_data.py |
|--------|----------------------|----------------------|
| `_fetch_available_model()` | L19-29 | L31-41 |
| `build_ebd_pairs()` | L222-254 | L237-266 |
| `build_levir_pairs()` | L257-275 | L269-283 |
| `build_second_pairs()` | L278-296 | L287-302 |
| `encode_image()` | L217-219 | L330-332 |
| `DATASET_ROOT` | L53 | L65 |
| `EBD_PER_EVENT/LEVIR_TAKE/SECOND_TAKE` | L57-59 | L72-74 |
| argparse 模板 | L33-43 | L45-54 |
| 断点续传逻辑 | `_load_completed` 内联 | `_load_completed` 函数 |
| 重试+指数退避 | L311-348 | `_call_with_retry` |

### 1.2 不一致之处

- **并发模型**：SFT 用 `ThreadPoolExecutor`（同步阻塞），VQA 用 `asyncio+Semaphore`（异步非阻塞）
- **API 客户端**：SFT 用 `OpenAI`（同步），VQA 用 `AsyncOpenAI`（异步）
- **重试策略**：SFT 在 `_process_single()` 内手写 for 循环，VQA 用 `_call_with_retry()` 包装
- **错误处理**：SFT 单条失败写 `[FAIL ...]` 到 stderr，VQA 记入 `fail_list`

### 1.3 耦合问题

`construct_vqa_data.py` 必须知道 SFT 的采样规则（`EBD_PER_EVENT=200`, `LEVIR_TAKE=500`, `SECOND_TAKE=2000`），并重新构建 pair 列表来计算 `get_cot_image_set()`，才能排除已占用的图像。两脚本各自维护一份相同的 pair 构建逻辑。

---

## 二、重构目标

1. **提取公共模块**：消除 ~200 行重复代码
2. **统一并发模型**：全部改用 asyncio
3. **解耦数据分配**：CoT 和 VQA 的分配策略集中在数据源层，不再各自复刻
4. **Prompt 外置**：从代码中剥离到独立文件，便于维护和版本管理
5. **路径可配置**：消除硬编码的 `DATASET_ROOT` 和 `OUTPUT_DIR`

---

## 三、提议的新架构

```
grpo/
├── data_builder/                   # 新建包
│   ├── __init__.py
│   ├── config.py                   # 全局配置：路径、采样参数
│   ├── sources.py                  # 数据源抽象层：EBD/LEVIR/SECOND 的 pair 构建 + 分配策略
│   ├── prompts.py                  # 所有 prompt 模板
│   ├── llm.py                      # LLM 调用封装：async chat + 重试 + 指数退避
│   ├── resume.py                   # 断点续传：加载已完成、加载 desc_map
│   ├── sft.py                      # SFT 数据集构造逻辑（图像对）
│   └── vqa.py                      # VQA 数据集构造逻辑（单图三阶段）
├── construct_sft_data.py           # 入口脚本，精简到 ~20 行
├── construct_vqa_data.py           # 入口脚本，精简到 ~20 行
└── augment_vqa_prompts.py          # 保持不变（独立功能）
```

### 3.1 各模块职责

**`config.py`** — 唯一配置来源
```python
DATASET_ROOT = "/home/charles/mycode/sft+rl/dataset"
SFT_OUTPUT_DIR = Path("/home/charles/mycode/grpo/sft_data")
VQA_OUTPUT_DIR = Path("/home/charles/mycode/grpo/vqa_data")

# 采样参数（只定义一次）
EBD_PER_EVENT = 200
LEVIR_TAKE = 500
SECOND_TAKE = 2000
VQA_MAX_PER_DATASET = 3000
```

**`sources.py`** — 数据源抽象
- 三个数据集的 pair 构建逻辑只写一次
- 提供 `build_pairs()` 返回所有 image pair
- 提供 `get_cot_allocation()` 返回 CoT 占用的 post 图集合
- 提供 `get_vqa_candidates()` 返回 VQA 可用 post 图列表
- 这样 VQA 脚本不再需要知道 CoT 的采样规则

关键设计：**分配策略集中管理**
```python
def get_cot_allocation() -> set[str]:
    """CoT 占用的 post 图集合"""
    ...

def get_vqa_candidates(max_per_dataset) -> list[tuple[str, str]]:
    """VQA 可用 post 图，返回 (img_path, dataset_name)"""
    ...
```

**`prompts.py`** — 所有 prompt 模板
- `EBD_TEACHER_PROMPT`, `LEVIR_TEACHER_PROMPT`, `SECOND_TEACHER_PROMPT`
- `DUAL_SCALE_PROMPT`, `Q1_GEN_PROMPT`, `Q2_GEN_PROMPT`
- 各数据集的 `USER_INSTRUCTIONS` 列表

**`llm.py`** — LLM 调用层
- `encode_image()` 函数
- `ChatClient` 类封装 AsyncOpenAI
  - `chat_with_image(prompt, image_path)` → str
  - `chat_text_only(prompt)` → str
  - 内置重试+指数退避
- `_fetch_available_model()` 函数

**`resume.py`** — 断点续传
- `load_completed(filepath) -> set[str]`：读取已完成图像的 `images[0]` 集合
- `load_desc_map(filepath) -> dict[str, str]`：读取 `{img_path: description}` 映射

**`sft.py`** — SFT 构建逻辑
- `build_sft_dataset(name, pairs, teacher_prompt, user_instructions, output_file, concurrency)`
- 内部用 `asyncio.Semaphore` 控制并发，每个图像对一次 API 调用生成分析报告

**`vqa.py`** — VQA 构建逻辑
- 三阶段顺序执行的 orchestration
- 每阶段内部并发调用

### 3.2 并发模型统一

全部改用 asyncio，理由：
- SFT 目前的 `ThreadPoolExecutor` 为每个请求阻塞一个线程，4 并发就是 4 个线程在 sleep 等待 HTTP 响应，不如 asyncio 的协程高效
- 两份脚本用同一套并发模式，减少心智负担
- AsyncOpenAI 已有成熟封装

### 3.3 数据格式（保持不变）

输出的 JSONL 格式不变，与 `dataset_info.json` 兼容：
```json
{
  "messages": [
    {"role": "user", "content": "...<image>..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["/abs/path/to/img.png"]
}
```

---

## 四、有争议/需要讨论的点

### Q1：Prompt 外置到什么程度？

**方案 A**：外置到 `.py` 文件（`prompts.py`），保持纯 Python 字符串。优点是不需要引入 YAML/JSON 解析，可直接用 `.format()`。

**方案 B**：外置到 YAML/JSON 文件。优点是纯数据，非程序员也能修改 prompt。但 prompt 中大量使用 `{}`（JSON 示例），与 `.format()` 冲突，需要额外的转义。

我的建议是 **方案 A**，理由是当前 prompt 中包含大量 `{}` 用于 JSON 示例和 `.format()` 占位符，放 YAML 中需要 `{{}}` 转义，反而容易出错。

### Q2：`augment_vqa_prompts.py` 要不要也重构？

这个脚本是离线操作：读取已有 JSONL，改写 `user` 角色的 `content` 再写回，不涉及 API 调用。与 SFT/VQA 构造逻辑正交，建议**保持不变**，只把文件路径改为从 `config.py` 引用。

### Q3：SFT 的 `user_instructions` 要不要也进入 VQA 风格的 teacher→student 流程？

当前 SFT 数据构造中，user prompt 是随机从 `USER_INSTRUCTIONS` 列表中选的简短模板（如"你现在是一名灾后评估专家..."），而 teacher model 收到的是详细的 `TEACHER_PROMPT`。VQA 中同样，user 看到的是简短的 `DESC_USER_INSTRUCTIONS`，teacher 收到的是 `DUAL_SCALE_PROMPT`。

这个设计是合理的（teacher 需要详细指导，student 数据需要多样化），保持不变。

### Q4：要不要合并两个入口脚本为一个？

**方案 A**：保持两个入口 `construct_sft_data.py` 和 `construct_vqa_data.py`。清晰独立，互不干扰。

**方案 B**：合并为一个 `build_all.py`，通过子命令区分。

建议 **方案 A**，两个任务差异大（两图 vs 单图，直接生成 vs 三阶段流水线），合并必要性不强。

### Q5：公共模块放在哪里？

**方案 A**：新建 `data_builder/` 包。干净，职责清晰，不污染根目录。

**方案 B**：新建 `common.py` 单文件。简单，文件少。

建议 **方案 A**，从项目角度看这是一个子系统，值得有独立命名空间。

---

## 五、实施步骤

1. 创建 `data_builder/` 目录结构
2. 先写 `config.py`，集中所有常量
3. 写 `llm.py`，封装 AsyncOpenAI + 重试
4. 写 `resume.py`，提取断点续传通用函数
5. 写 `sources.py`，只写一次的 pair 构建 + 分配逻辑
6. 写 `prompts.py`，搬运所有 prompt 字符串
7. 写 `sft.py`，用 asyncio 重写 SFT 构造
8. 写 `vqa.py`，搬运 VQA 三阶段逻辑（主要是 orchestration，底层调用已在 llm.py）
9. 改写两个入口脚本，精简到 20 行
10. 更新 `print_samples.py` 和 `augment_vqa_prompts.py` 中的硬编码路径

---

## 六、预期收益

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 代码总行数 | ~1400（两个脚本） | ~900（公共300 + sft 250 + vqa 300 + 入口40） |
| 重复的 pair 构建逻辑 | 3 个数据集 × 2 脚本 = 6 处 | 1 处（sources.py） |
| 并发模型 | 两种（ThreadPool + asyncio） | 一种（asyncio） |
| 路径硬编码 | 2 处 DATASET_ROOT | 1 处 config.py |
| Prompt 维护 | 散落在脚本中间位置 | 集中在 prompts.py |
| CoT/VQA 分配耦合 | VQA 脚本复刻 CoT 规则 | sources.py 统一提供 |

---

请针对以上方案提出你的意见，特别是 Q1-Q5 几个需要决策的点。确认方向后我们开始改代码。
