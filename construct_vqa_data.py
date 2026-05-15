"""
构造双尺度描述 + VQA 数据集。

数据流:
  单张遥感图 → 教师模型生成双尺度描述（宏观场景 + 微观地物）
            → 基于描述生成 Q1（开放式）和 Q2（空间关系/计数）问答对

输出三个 JSONL:
  - dual_scale_desc.jsonl   : 图像 → 双尺度描述（SFT 格式）
  - vqa_openended.jsonl     : 图像 → Q1 开放式问答（SFT 格式）
  - vqa_specific.jsonl      : 图像 → Q2 具体关系/计数问答（SFT 格式）
"""

import base64
import json
import random
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ==================== 配置 ====================
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
model_name = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_ROOT = "/home/charles/mycode/sft+rl/dataset"
OUTPUT_DIR = Path("/home/charles/mycode/grpo/vqa_data")

# 每个数据集采样上限（从 CoT 剩余中采样）
MAX_PER_DATASET = 3000

# ==================== CoT 分配规则（与 construct_sft_data.py 保持一致） ====================
EBD_PER_EVENT = 200       # EBD 每种灾害类型前 200 对归 CoT
LEVIR_TAKE = 500           # LEVIR-CD+ 前 500 对归 CoT，剩余归 VQA
SECOND_TAKE = 2000         # SECOND 前 2000 对归 CoT

# ==================== Prompt: 双尺度描述生成 ====================

DUAL_SCALE_PROMPT = """# Role
你是一位顶级的遥感图像解译专家，同时具备宏观地理学视野与微观地物判读能力。

# Task
请对这张遥感图像进行**双尺度解译**，严格按以下两层结构输出。你的描述将作为后续自动生成问答对的知识源，因此必须信息密集、具体、可验证。

---

## 一、宏观场景层

### 1.1 场景类型
判定图像所属的场景类别（如：城市建成区 / 乡村聚落 / 工业仓储区 / 农田种植区 / 水域湿地 / 山地森林 / 交通枢纽 / 灾后现场 等），可多类别叠加。

### 1.2 空间布局
- 描述整体空间结构（如：网格状路网、沿河带状分布、核心-外围扩散、散点式分布等）。
- 划分主要功能分区，说明各分区的方位与大致占比（如："图像上部约 1/3 为密集居民区，下部为开阔农田"）。
- 描述主要线性地物的走向与连通关系（道路、河流、铁路、管线等）。

### 1.3 景观格局
- 描述斑块大小特征（如：大面积连片 / 细碎镶嵌）。
- 判断连通性（如：廊道贯通 / 斑块隔离）。
- 评估人工化程度（如：高度人工化 / 半自然 / 近自然）。

---

## 二、微观地物层

### 2.1 地物清单
逐一列出图像中可辨识的具体地物实例。每个实例用独立条目描述，格式如下：
- **[类别] 简要标识**：位于<方位>。视觉特征：<颜色>、<形状>、<尺寸估计>、<纹理/材质>。

覆盖以下类别（至少覆盖 5 类，每类至少 2 个实例）：
- 建筑物（注明颜色、层数估计、屋顶类型）
- 车辆（注明颜色、类型如轿车/卡车/工程车）
- 植被（注明类型如乔木/灌木/草地/农田、颜色深浅反映的生长状态）
- 道路（注明路面颜色、宽度估计、车道数估计）
- 水体（注明颜色、形态如线状/面状）
- 其他人工设施（围墙、棚架、储罐、电线杆、船舶等）

### 2.2 地物空间关系
列出不少于 5 条具体的地物间空间关系。每条格式：
- <地物A> 的 <方位> 有 <数量> 个/辆/栋 <地物B>
- <地物A> 与 <地物B> 之间隔着 <地物C>
- <地物A> 位于 <地物B> 的 <方位>，距离约 <估计>
- <地物A> 周边分布着 <地物B> 和 <地物C>

### 2.3 可计数清单
单独列出一个"可计数清单"，格式为：
- <类别>: <数量> <单位>（如：红色屋顶建筑: 3 栋、白色轿车: 5 辆）

---

# Output Rules
1. 使用中文输出。
2. 不确定处用"疑似""约""估计"。
3. 禁止开场白、结束语，直接输出分析内容。
4. 描述必须足够具体，使得后续可以不看图仅凭文字回答关于此图的细节问题。
"""

# ==================== Prompt: Q1 开放式问答生成 ====================

Q1_GEN_PROMPT = """# Role
你是一位遥感图像 VQA 数据标注专家。

# Context
下面是一张遥感图像的**双尺度描述文本**（包含宏观场景信息和微观地物细节）。
你的任务是基于这段描述，生成**开放式总体理解问题**及其标准答案。

# Task
基于提供的描述，生成 {num_q} 个开放式问题。问题类型应多样化，从以下类型中均匀选取：

**问题类型：**
1. **场景总览**："这张遥感图像中有什么？""请描述这张图像的内容。"
2. **场景分类**："这是什么类型的场景？""这个区域的主要功能是什么？"
3. **空间结构**："这个区域的空间布局是怎样的？""图像中有哪些主要的功能分区？"
4. **地物构成**："图像中有哪些类型的地物？""该区域包含哪些土地利用类型？"
5. **综合特征**："这个区域最显著的特征是什么？""该场景的人工化程度如何？"
6. **景观描述**："描述这个区域的植被覆盖情况。""这个区域的水体分布有什么特点？"

**答案要求：**
- 答案必须基于描述文本，不可编造描述中没有的信息。
- 答案应充分、完整，长度在 50-200 字。
- 对于概览类问题，答案应涵盖宏观和微观两个层面的信息。

# Input: 图像双尺度描述
{description}

# Output Format
输出一个 JSON 数组（不要包含其他文字）：
```json
[
  {{"question": "问题文本", "answer": "答案文本"}},
  ...
]
```
"""

# ==================== Prompt: Q2 具体关系/计数问答生成 ====================

Q2_GEN_PROMPT = """# Role
你是一位遥感图像 VQA 数据标注专家。

# Context
下面是一张遥感图像的**双尺度描述文本**（包含宏观场景信息和微观地物细节）。
你的任务是基于这段描述，生成**具体的空间关系/属性查询/计数类问题**及其标准答案。

# Task
基于提供的描述，生成 {num_q} 个具体问题。问题类型应多样化，从以下类型中均匀选取：

**问题类型：**
1. **空间关系查询**："<颜色/属性>的<地物A>旁边有什么？""<地物A>的<方位>是什么？"
2. **计数问题**："图中有几辆<颜色>的<地物>？""<区域>有多少栋<属性>建筑？"
3. **属性查询**："<地物>是什么颜色的？""<地物>的屋顶是什么类型？"
4. **方位定位**："<地物A>在<地物B>的哪个方向？""<地物>位于图像的什么位置？"
5. **比较问题**："<区域A>和<区域B>之间有什么？""<地物A>和<地物B>哪个更大/更多？"
6. **存在性判断**："图中有没有<地物>？""<区域>是否有<植被/水体/建筑>？"

**问题设计要求：**
- 每个问题必须包含至少一个**具体的视觉属性约束**（颜色、形状、大小、材质、方位等），如"红色的房子""左侧的河流""最大的建筑"。
- 计数类问题的数量必须是描述中明确可数的。
- 空间关系问题必须涉及至少两个地物之间的方位、距离或拓扑关系。
- 答案必须精确、简短、可以从描述中直接验证。计数类答案纯数字即可（如"3 辆"），空间类答案简洁明确（如"东侧""紧邻"）。

# Input: 图像双尺度描述
{description}

# Output Format
输出一个 JSON 数组（不要包含其他文字）：
```json
[
  {{"question": "问题文本", "answer": "答案文本"}},
  ...
]
```
"""

# ==================== 用户指令（多样化，存入 SFT 数据） ====================

DESC_USER_INSTRUCTIONS = [
    "请对这张遥感图像进行双尺度解译，覆盖宏观场景逻辑与微观地物特征。",
    "从宏观场景和微观地物两个层面，分析这张遥感图像。",
    "帮我看看这张遥感图里有什么，既要描述整体场景，也要列出具体的地物细节。",
    "请对这张遥感影像进行全面解译，包括场景类型判断和具体地物识别。",
    "你是一名遥感解译专家，请从整体布局到局部细节分析这张图像。",
    "分析这张遥感图：先描述大场景（是什么地方、怎么布局），再识别小地物（建筑、车辆、植被等）。",
    "请解译这张遥感图像，输出宏观场景逻辑与微观地物特征两个层次的信息。",
    "看看这张遥感影像，告诉我这是什么场景，里面有哪些具体的东西。",
    "对这张图像进行遥感解译，覆盖场景分类、空间布局、地物清单和空间关系。",
    "请从宏观（场景类型、空间结构）和微观（地物属性、空间关系）两个尺度描述这张图。",
    "作为遥感图像分析专家，请对这张图进行双层次解译：先整体后局部。",
    "描述这张遥感图像的内容，要求既有大局观（场景、布局）又有细节（具体地物、位置关系）。",
]


# ==================== CoT 图像占用计算 ====================

def _build_ebd_pairs(root_path):
    """构建 EBD 数据集 pre/post 对（与 construct_sft_data.py 一致）。"""
    root = Path(root_path)
    pairs = []
    for event_dir in root.iterdir():
        if not event_dir.is_dir():
            continue
        event = event_dir.name
        img_dir = event_dir / "images"
        if not img_dir.exists():
            continue
        id_to_paths = defaultdict(dict)
        for img_path in img_dir.glob("*"):
            stem = img_path.stem
            parts = stem.rsplit("_", 2)
            if len(parts) < 3:
                continue
            img_id = parts[-2]
            if "pre" in stem:
                id_to_paths[img_id]["pre"] = img_path
            elif "post" in stem:
                id_to_paths[img_id]["post"] = img_path
        for img_id, paths in id_to_paths.items():
            if "pre" in paths and "post" in paths:
                pairs.append({
                    "event": event,
                    "pre": paths["pre"],
                    "post": paths["post"],
                })
    return pairs


def _build_levir_pairs(root_path):
    """构建 LEVIR-CD+ 数据集 A/B 对（与 construct_sft_data.py 一致）。"""
    root = Path(root_path)
    pairs = []
    for split in ["train", "test"]:
        dir_a = root / split / "A"
        dir_b = root / split / "B"
        if not dir_a.exists() or not dir_b.exists():
            continue
        for img_path in sorted(dir_a.iterdir()):
            if not img_path.is_file():
                continue
            counterpart = dir_b / img_path.name
            if counterpart.exists():
                pairs.append({"pre": img_path, "post": counterpart})
    return pairs


def _build_second_pairs(root_path):
    """构建 SECOND 数据集 im1/im2 对（与 construct_sft_data.py 一致）。"""
    root = Path(root_path)
    pairs = []
    for split in ["train", "test"]:
        dir_im1 = root / split / "im1"
        dir_im2 = root / split / "im2"
        if not dir_im1.exists() or not dir_im2.exists():
            continue
        for img_path in sorted(dir_im1.iterdir()):
            if not img_path.is_file():
                continue
            counterpart = dir_im2 / img_path.name
            if counterpart.exists():
                pairs.append({"pre": img_path, "post": counterpart})
    return pairs


def get_cot_image_set():
    """返回 CoT 数据占用的图像路径集合（绝对路径字符串）。
    分配规则与 construct_sft_data.py 完全一致：
      - EBD: 每种事件前 EBD_PER_EVENT 对
      - LEVIR-CD+: 前 LEVIR_TAKE 对
      - SECOND: 前 SECOND_TAKE 对
    """
    cot_images = set()

    # EBD
    ebd_all = _build_ebd_pairs(DATASET_ROOT + "/EBD")
    ebd_by_event = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    for event, pairs in sorted(ebd_by_event.items()):
        for p in pairs[:EBD_PER_EVENT]:
            cot_images.add(str(p["pre"].resolve()))
            cot_images.add(str(p["post"].resolve()))

    # LEVIR-CD+: 前 LEVIR_TAKE 对
    for p in _build_levir_pairs(DATASET_ROOT + "/LEVIR-CD+")[:LEVIR_TAKE]:
        cot_images.add(str(p["pre"].resolve()))
        cot_images.add(str(p["post"].resolve()))

    # SECOND: 前 SECOND_TAKE 对
    second_all = _build_second_pairs(DATASET_ROOT + "/SECOND")
    for p in second_all[:SECOND_TAKE]:
        cot_images.add(str(p["pre"].resolve()))
        cot_images.add(str(p["post"].resolve()))

    return cot_images


# ==================== 工具函数 ====================


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_unique_images(root_path, dataset_type):
    """从各数据集中收集唯一图像路径（已 resolve，与 get_cot_image_set 一致）。"""
    root = Path(root_path)
    images = set()

    if dataset_type == "ebd":
        for event_dir in root.iterdir():
            if not event_dir.is_dir():
                continue
            img_dir = event_dir / "images"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    images.add(str(img_path.resolve()))
    elif dataset_type in ("levir", "second"):
        for split in ["train", "test"]:
            for subdir in ["A", "B"] if dataset_type == "levir" else ["im1", "im2"]:
                d = root / split / subdir
                if not d.exists():
                    continue
                for img_path in d.iterdir():
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                        images.add(str(img_path.resolve()))

    return sorted(images)


def chat_with_image(prompt, image_path, temperature=0.7, max_tokens=2048):
    """调用 VLM 处理单张图像。"""
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def chat_text_only(prompt, temperature=0.8, max_tokens=2048):
    """纯文本调用 LLM（用于从描述生成 QA，不需要图像）。"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def extract_json_array(text):
    """从文本中提取 JSON 数组。"""
    # 尝试直接解析
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 数组边界
    for match in re.finditer(r"\[.*\]", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


# ==================== 主程序 ====================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    desc_file = OUTPUT_DIR / "dual_scale_desc.jsonl"
    q1_file = OUTPUT_DIR / "vqa_openended.jsonl"
    q2_file = OUTPUT_DIR / "vqa_specific.jsonl"

    # ---- 计算 CoT 已占用图像 ----
    print("计算 CoT 数据占用的图像...")
    cot_images = get_cot_image_set()
    print(f"  CoT 共占用: {len(cot_images)} 张图像\n")

    # ---- 收集图像（排除 CoT 已占用） ----
    print("收集 VQA 可用图像（排除 CoT 已占用）...")
    all_images = []
    for dtype, ds_name in [("ebd", "EBD"), ("levir", "LEVIR-CD+"), ("second", "SECOND")]:
        ds_path = DATASET_ROOT + "/" + ds_name
        imgs = collect_unique_images(ds_path, dtype)
        available = [p for p in imgs if p not in cot_images]
        sampled = available[:MAX_PER_DATASET]
        print(f"  {ds_name}: {len(imgs)} 张, CoT占用 {len(imgs)-len(available)}, VQA可用 {len(available)}, 采样 {len(sampled)}")
        all_images.extend(sampled)

    random.shuffle(all_images)
    print(f"  总计: {len(all_images)} 张图像\n")

    # ---- 断点续传：加载已完成 ----
    done_desc = set()
    for f in [desc_file, q1_file, q2_file]:
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done_desc.add(rec["images"][0])
                    except Exception:
                        pass

    remaining = [p for p in all_images if p not in done_desc]
    print(f"已完成: {len(done_desc)}, 待处理: {len(remaining)}\n")

    if not remaining:
        print("全部已完成。")
        return

    fail_list = []

    with (
        open(desc_file, "a", encoding="utf-8") as f_desc,
        open(q1_file, "a", encoding="utf-8") as f_q1,
        open(q2_file, "a", encoding="utf-8") as f_q2,
    ):
        for img_path in tqdm(remaining, desc="生成双尺度描述+VQA"):
            try:
                # Step 1: 生成双尺度描述
                description = chat_with_image(DUAL_SCALE_PROMPT, img_path, temperature=0.7, max_tokens=2048)

                if not description or len(description.strip()) < 50:
                    print(f"\n[跳过] {Path(img_path).name}: 描述过短")
                    continue

                # 写入描述数据
                desc_record = {
                    "messages": [
                        {"role": "user", "content": f"<image>\n{random.choice(DESC_USER_INSTRUCTIONS)}"},
                        {"role": "assistant", "content": description.strip()},
                    ],
                    "images": [img_path],
                }
                f_desc.write(json.dumps(desc_record, ensure_ascii=False) + "\n")
                f_desc.flush()

                # Step 2: 基于描述生成 Q1（开放式问答）
                q1_prompt = Q1_GEN_PROMPT.format(num_q=4, description=description)
                q1_raw = chat_text_only(q1_prompt, temperature=0.8, max_tokens=2048)
                q1_pairs = extract_json_array(q1_raw)

                if q1_pairs:
                    for pair in q1_pairs:
                        q1_record = {
                            "messages": [
                                {"role": "user", "content": f"<image>\n{pair['question']}"},
                                {"role": "assistant", "content": pair["answer"]},
                            ],
                            "images": [img_path],
                        }
                        f_q1.write(json.dumps(q1_record, ensure_ascii=False) + "\n")
                    f_q1.flush()
                else:
                    print(f"\n[Q1解析失败] {Path(img_path).name}: {q1_raw[:200]}")

                # Step 3: 基于描述生成 Q2（具体关系/计数问答）
                q2_prompt = Q2_GEN_PROMPT.format(num_q=5, description=description)
                q2_raw = chat_text_only(q2_prompt, temperature=0.8, max_tokens=2048)
                q2_pairs = extract_json_array(q2_raw)

                if q2_pairs:
                    for pair in q2_pairs:
                        q2_record = {
                            "messages": [
                                {"role": "user", "content": f"<image>\n{pair['question']}"},
                                {"role": "assistant", "content": pair["answer"]},
                            ],
                            "images": [img_path],
                        }
                        f_q2.write(json.dumps(q2_record, ensure_ascii=False) + "\n")
                    f_q2.flush()
                else:
                    print(f"\n[Q2解析失败] {Path(img_path).name}: {q2_raw[:200]}")

            except KeyboardInterrupt:
                print(f"\n用户中断。")
                return
            except Exception as e:
                print(f"\n[错误] {Path(img_path).name}: {e}")
                fail_list.append(img_path)

    # ---- 统计 ----
    print("\n" + "=" * 60)
    print("完成！输出文件：")
    for label, f in [("双尺度描述", desc_file), ("Q1-开放式VQA", q1_file), ("Q2-具体关系VQA", q2_file)]:
        if f.exists():
            count = sum(1 for _ in open(f))
            print(f"  {label}: {f}  ({count} 条)")
    if fail_list:
        print(f"\n失败 {len(fail_list)} 条（下次运行自动重试）")
    print("=" * 60)


if __name__ == "__main__":
    main()
