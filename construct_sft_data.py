import argparse
import base64
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from urllib.request import Request, urlopen

from openai import OpenAI
from tqdm import tqdm


def _fetch_available_model(base_url: str) -> str:
    req = Request(f"{base_url}/v1/models", headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return ""


# ==================== 配置 ====================
parser = argparse.ArgumentParser(description="Construct SFT data from teacher model")
parser.add_argument("--model", default=None,
                    help="Model name (default: $VLLM_MODEL or auto-detect from server)")
parser.add_argument("--base-url", default="http://localhost:8001",
                    help="vLLM server base URL (default: http://localhost:8001)")
parser.add_argument("--concurrency", type=int, default=4,
                    help="Number of parallel requests to vLLM (default: 4)")
parser.add_argument("--retries", type=int, default=3,
                    help="Max retries per record on failure (default: 3)")
args = parser.parse_args()

base_url = args.base_url
model_name = args.model or os.getenv("VLLM_MODEL") or _fetch_available_model(base_url)
if not model_name:
    print("FAIL: unable to determine model. Pass --model or set $VLLM_MODEL.")
    sys.exit(1)
print(f"Using model: {model_name}")

client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")

DATASET_ROOT = "/home/charles/mycode/sft+rl/dataset"
OUTPUT_DIR = Path("/home/charles/mycode/grpo")

# 每个数据集的采样数量
EBD_PER_EVENT = 200  # EBD 每种灾害类型抽 200 对
LEVIR_TAKE = 500  # LEVIR-CD+ 按顺序取前 500 对
SECOND_TAKE = 2000  # SECOND 按顺序取前 2000 对

# ==================== 教师模型 Prompt（详细，发给教师模型） ====================

EBD_TEACHER_PROMPT = """# Role
你是一位资深的遥感地质专家与防灾减灾顾问。请对比分析提供的两张影像（图1灾前，图2灾后）。

# Task
请对比分析上传的两张图像，并严格按照以下结构输出技术报告：

## 1. 灾害类型判定与物理机制
- **判定结论**：给出具体的灾害名称（如：走滑断层地表破裂、推覆构造、浅表层滑坡等）。
- **特征描述**：结合影像中的光谱特征、纹理断裂及几何位移，描述灾害表现（如：地表破裂线走向、地表粗糙度变化、地物水平位移量估算）。
- **成因简析**：简述地质或气象驱动机制。

## 2. 针对性损害评估
- **基础设施**：指明具体受损点（如：道路某段的物理偏移、桥梁坍塌、电力塔基变形）。
- **农业与生态**：分析具体斑块的受损情况（如：林地破碎化、农田灌溉渠断裂、生态斑块连通性受阻）。
- **建筑安全**：针对影像中具体建筑的基底偏移、倒塌或疑似结构损伤进行说明。

## 3. 差异化防治建议
- **工程避让**：根据破裂线或灾害影响范围，划定精确的建筑禁建区或缓冲区。
- **韧性修复**：针对影像中损坏的具体设施提供修复方案（如：跨断层的柔性管道连接、加筋土路基复原）。
- **监测布控**：建议在影像中受损严重的特定坐标点布设形变监测传感器。

# Output Style (严格执行)
1. **专业性**：必须使用灾害分析术语（如：地表形变矢量、生境破碎化、光谱差异、构造应力）。
2. **客观性**：严禁过度推测。若影像模糊，必须使用"疑似"、"迹象显示"或"推测可能"等词汇。
3. **简洁性**：直接输出分析内容，禁止输出任何开场白、解释性文字或结束语（如"好的"、"我为您分析"等）。
4. **针对性**：所有分析必须锁定图像中的具体地物（如：指明"影像中部的矩形耕地"、"右侧的红色屋顶建筑"），拒绝通用化模板。"""

LEVIR_TEACHER_PROMPT = """# Context
你正在分析两幅时序遥感影像（图1为早期影像，图2为后期影像），场景以城市建筑物为主。

# Role
你是一位资深的遥感图像解译与城市规划专家。请对比分析提供的两张影像，识别并评估建筑物变化情况。

# Task
请对比分析上传的两张图像，并严格按照以下结构输出变化检测报告：

## 1. 建筑物变化总体概述
- **变化规模**：概述两张影像之间建筑物的总体变化程度（如：大范围新建、局部拆除、少量改建等）。
- **空间分布**：描述变化发生的空间分布特征（如：集中在影像上部、沿道路两侧、分散分布等）。

## 2. 新增建筑物分析
- **具体位置**：锁定影像中新增建筑物的具体位置（如："影像左上角出现了一栋白色矩形建筑"）。
- **建筑特征**：描述新增建筑的几何形态、光谱特征、规模估算。
- **用地性质推测**：根据建筑形态和周边环境，推测可能的用地性质（居住、商业、工业等）。

## 3. 拆除/消失建筑物分析
- **具体位置**：锁定影像中建筑物消失或拆除的具体位置。
- **变化特征**：描述原建筑位置当前的地表状态（如：已恢复为裸土、被植被覆盖、改建为道路等）。

## 4. 改建/扩建建筑物分析
- **具体位置**：锁定影像中建筑形态发生明显变化的位置。
- **变化细节**：描述建筑扩建、加层、结构改造等具体变化特征。

## 5. 周边环境影响评估
- **道路与交通**：分析建筑物变化对周边路网的影响。
- **植被与空地**：分析建筑物变化导致的植被覆盖或空地变化。
- **城市扩张趋势**：综合判断该区域的城市发展方向与趋势。

# Output Style (严格执行)
1. **专业性**：必须使用遥感解译术语（如：光谱特征、几何形态、空间纹理、归一化建筑指数等）。
2. **客观性**：严禁过度推测。若影像模糊，必须使用"疑似"、"迹象显示"或"推测可能"等词汇。
3. **简洁性**：直接输出分析内容，禁止输出任何开场白、解释性文字或结束语。
4. **针对性**：所有分析必须锁定图像中的具体地物位置，拒绝通用化模板。"""

SECOND_TEACHER_PROMPT = """# Context
你正在分析两幅时序遥感影像（图1为早期影像，图2为后期影像），场景覆盖多种地表类型。

# Role
你是一位资深的遥感图像解译与生态环境监测专家。请对比分析提供的两张影像，识别并评估地表覆盖变化情况。

# Task
请对比分析上传的两张图像，并严格按照以下结构输出变化检测报告：

## 1. 地表覆盖变化总体概述
- **变化规模**：概述两张影像之间地表覆盖的总体变化程度（如：大范围植被退化、局部水体扩张、城市用地增加等）。
- **变化类型**：列出影像中存在的主要变化类型（如：植被→裸土、水体→陆地、农田→建筑等）。

## 2. 植被覆盖变化
- **植被增加区域**：锁定影像中植被覆盖增加的斑块位置，描述增加的范围与可能的成因（自然恢复、人工造林等）。
- **植被减少区域**：锁定影像中植被退化的斑块位置，描述减少的范围与可能的成因（砍伐、火灾、干旱等）。
- **植被类型转换**：识别植被类型发生变化的区域（如：林地→草地、湿地→旱地等）。

## 3. 水体变化
- **水体扩张**：锁定影像中水体面积增加的斑块位置，分析可能的成因（洪水、水库蓄水、海平面上升等）。
- **水体萎缩**：锁定影像中水体面积减少的斑块位置，分析可能的成因（干旱、填湖造陆、过度取水等）。
- **水质变化迹象**：若影像中水体颜色/光谱特征发生明显变化，描述并推测可能的原因。

## 4. 建设用地变化
- **新增建设用地**：锁定影像中新增的建设用地斑块位置，描述规模和形态特征。
- **道路与基础设施**：识别新增或扩建的道路、桥梁等基础设施。
- **工业/矿区变化**：识别工矿用地的扩张或废弃迹象。

## 5. 生态环境影响评估
- **生态连通性**：分析植被变化的斑块连通性影响。
- **土壤退化风险**：识别裸土扩张区域，评估土壤侵蚀或荒漠化风险。
- **综合趋势判断**：综合判断该区域的生态环境演变趋势。

# Output Style (严格执行)
1. **专业性**：必须使用遥感与生态学术语（如：归一化植被指数特征、斑块连通性、光谱反射率变化、生态系统服务功能等）。
2. **客观性**：严禁过度推测。若影像模糊，必须使用"疑似"、"迹象显示"或"推测可能"等词汇。
3. **简洁性**：直接输出分析内容，禁止输出任何开场白、解释性文字或结束语。
4. **针对性**：所有分析必须锁定图像中的具体地物位置，拒绝通用化模板。"""

# ==================== SFT 用户指令（多样化，存入 SFT 数据） ====================

EBD_USER_INSTRUCTIONS = [
    "你现在是一名灾后评估专家。请通过这两张对比图分析受灾情况并给出建议。",
    "作为灾害评估专家，请对比这两张灾前灾后遥感影像，分析灾害类型与损失情况。",
    "请以地质灾害调查员的身份，对比分析这两张图像中的灾害特征。",
    "你需要对这两张遥感图像进行灾前灾后对比，评估灾害造成的损害程度。",
    "对比这两张时序影像，识别灾害类型、分析损害范围，并给出防治建议。",
    "观察下面灾前和灾后两张遥感图，请进行灾害识别与损失评估。",
    "你是一名地质灾后评估顾问，请基于这两张对比图写一份灾害分析。",
    "请分析这两张遥感影像，判断发生了什么类型的灾害及其影响。",
    "看看这两张图（灾前/灾后），帮我分析一下灾害情况和应对措施。",
    "作为遥感解译专家，请对比这两张影像，完成灾害评估报告。",
    "请从灾害类型、损害程度、防治建议三个方面分析这两张对比图。",
    "对比这两张图像，告诉我这里发生了什么灾害，造成了哪些破坏。",
]

LEVIR_USER_INSTRUCTIONS = [
    "你现在是一名遥感变化检测专家。请通过这两张对比影像分析建筑物变化情况。",
    "作为遥感解译专家，请对比这两张不同时期的影像，检测建筑物的新增、拆除和改建。",
    "请分析这两张遥感图像，识别哪些区域发生了建筑物变化。",
    "对比这两张时序影像，找出建筑物变化的位置并描述变化类型。",
    "你是一名城市规划分析师，请对比这两张图，检测建筑物的变化。",
    "观察这两张不同时期的遥感影像，分析建筑物发生了哪些变化。",
    "请对这两张图像进行建筑物变化检测，描述新增、消失和改建的具体位置。",
    "看看这两张图，帮我找出建筑有变化的地方，并分析变化情况。",
    "作为建筑变化检测专家，请对比这两张影像，完成变化分析报告。",
    "对比这两张遥感图，识别建筑物变化区域并评估影响。",
    "请从新增建筑、拆除建筑、改建建筑三个方面分析这两张对比图。",
    "分析这两张前后时相的遥感影像，输出建筑物变化检测结果。",
]

SECOND_USER_INSTRUCTIONS = [
    "你现在是一名遥感变化检测专家。请通过这两张对比影像分析地表覆盖变化情况。",
    "作为生态环境监测专家，请对比这两张影像，分析地表覆盖类型的变化。",
    "请分析这两张不同时期的遥感图像，识别植被、水体、建设用地等的变化。",
    "对比这两张时序影像，分析地表覆盖发生了哪些转变。",
    "你是一名土地利用变化分析师，请对比这两张图，检测地表覆盖变化。",
    "观察这两张遥感影像，分析植被覆盖、水体和建设用地的变化情况。",
    "请对这两张图像进行地表覆盖变化检测，描述各类地物的变化。",
    "看看这两张不同时期的图，帮我分析地表覆盖有哪些变化。",
    "作为生态遥感专家，请对比这两张影像，完成地表覆盖变化分析。",
    "对比这两张遥感图，识别植被退化、水体变化和建设用地扩张。",
    "请从植被变化、水体变化、建设用地变化三个方面分析这两张对比图。",
    "分析这两张前后时相的遥感影像，评估地表覆盖变化及其生态影响。",
]


# ==================== 工具函数 ====================


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_ebd_pairs(root_path):
    """构建 EBD 数据集 pre/post 对，包含 event 信息。"""
    root = Path(root_path)
    pairs = []
    for event_dir in root.iterdir():
        if not event_dir.is_dir():
            continue
        event = event_dir.name
        img_dir = event_dir / "images"
        if not img_dir.exists():
            continue
        # 按 ID 分组
        id_to_paths = defaultdict(dict)
        for img_path in img_dir.glob("*"):
            # 格式: {EVENT}_{ID}_{pre/post}_disaster.{ext}
            stem = img_path.stem
            parts = stem.rsplit("_", 3)
            if len(parts) < 4:
                continue
            img_id = parts[-3]
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


def build_levir_pairs(root_path):
    """构建 LEVIR-CD+ 数据集 A/B 对。"""
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
                pairs.append({
                    "pre": img_path,
                    "post": counterpart,
                })
    return pairs


def build_second_pairs(root_path):
    """构建 SECOND 数据集 im1/im2 对。"""
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
                pairs.append({
                    "pre": img_path,
                    "post": counterpart,
                })
    return pairs


# ==================== 主程序 ====================


def _process_single(task, teacher_prompt_template, user_instructions):
    """处理单条记录，带重试逻辑。成功返回 record，失败返回 None。"""
    pre_path = str(task["pre"])
    b64_pre = encode_image(task["pre"])
    b64_post = encode_image(task["post"])

    max_retries = args.retries
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": teacher_prompt_template},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_pre}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_post}"}},
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.7,
            )

            answer = response.choices[0].message.content
            return {
                "messages": [
                    {"role": "user", "content": f"<image>\n<image>\n{random.choice(user_instructions)}"},
                    {"role": "assistant", "content": answer},
                ],
                "images": [str(task["pre"].resolve()), str(task["post"].resolve())],
            }

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)

    stem = Path(pre_path).stem
    tqdm.write(f"\n[FAIL {stem}] retries exhausted: {last_error}")
    return None


def process_dataset(name, pairs, teacher_prompt_template, user_instructions, output_file):
    """通用数据集处理函数，并行请求 vLLM，带重试和断点续传。"""
    output_path = Path(output_file)

    # 加载已完成记录
    done = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["images"][0])
                except Exception:
                    pass

    # 过滤出待处理的 pairs
    remaining = [t for t in pairs if str(t["pre"]) not in done]
    skipped = len(pairs) - len(remaining)

    print(f"\n{'=' * 60}")
    print(f"处理 {name}")
    print(f"  总计: {len(pairs)} 对 | 已完成: {skipped} | 待处理: {len(remaining)}")
    print(f"  输出: {output_file}")
    print(f"{'=' * 60}")

    if not remaining:
        print("  全部已完成，跳过。")
        return

    fail_list = []
    write_lock = threading.Lock()
    running = True

    with open(output_path, "a", encoding="utf-8") as f_out:
        with tqdm(total=len(remaining), desc=f"{name}") as pbar:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(_process_single, task, teacher_prompt_template, user_instructions): task
                    for task in remaining
                }

                try:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            record = future.result()
                        except Exception as e:
                            if not running:
                                break
                            stem = Path(task["pre"]).stem
                            tqdm.write(f"\n[错误] {name} | {stem}: {e}")
                            fail_list.append(str(task["pre"]))
                            pbar.update(1)
                            continue

                        if record is not None:
                            with write_lock:
                                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                f_out.flush()
                        else:
                            fail_list.append(str(task["pre"]))
                        pbar.update(1)

                except KeyboardInterrupt:
                    running = False
                    remaining_count = sum(1 for f in futures if not f.done())
                    print(f"\n用户中断，取消剩余 {remaining_count} 个任务...")
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    if fail_list:
                        print(f"本次失败 {len(fail_list)} 条，下次运行会自动重试。")
                    return

    done_count = len(remaining) - len(fail_list) - pbar.n + pbar.total
    if fail_list:
        print(f"\n{name} 处理完成，成功 {pbar.n - len(fail_list)} 条，失败 {len(fail_list)} 条（下次运行会自动重试）：")
        for fp in fail_list[:10]:
            print(f"  - {fp}")
        if len(fail_list) > 10:
            print(f"  ... 及其他 {len(fail_list) - 10} 条")


def main():
    # ── 构建所有数据集并打印汇总 ──
    ebd_all = build_ebd_pairs(DATASET_ROOT + "/EBD")
    ebd_by_event = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    ebd_pairs = []
    for event, pairs in sorted(ebd_by_event.items()):
        taken = pairs[:EBD_PER_EVENT]
        ebd_pairs.extend(taken)

    levir_all = build_levir_pairs(DATASET_ROOT + "/LEVIR-CD+")
    levir_pairs = levir_all[:LEVIR_TAKE]

    second_all = build_second_pairs(DATASET_ROOT + "/SECOND")
    second_pairs = second_all[:SECOND_TAKE]

    print("数据集统计：")
    for event, pairs in sorted(ebd_by_event.items()):
        print(f"  EBD | {event}: {len(pairs)} 对可用, 取前 {min(len(pairs), EBD_PER_EVENT)} 对")
    print(f"  LEVIR-CD+: {len(levir_all)} 对可用, 取前 {len(levir_pairs)} 对")
    print(f"  SECOND:    {len(second_all)} 对可用, 取前 {len(second_pairs)} 对")
    print(f"  合计待处理: {len(ebd_pairs) + len(levir_pairs) + len(second_pairs)} 对")

    # ── 依次处理 ──
    process_dataset(
        name="EBD",
        pairs=ebd_pairs,
        teacher_prompt_template=EBD_TEACHER_PROMPT,
        user_instructions=EBD_USER_INSTRUCTIONS,
        output_file=str(OUTPUT_DIR / "EBD_sft.jsonl"),
    )
    process_dataset(
        name="LEVIR-CD+",
        pairs=levir_pairs,
        teacher_prompt_template=LEVIR_TEACHER_PROMPT,
        user_instructions=LEVIR_USER_INSTRUCTIONS,
        output_file=str(OUTPUT_DIR / "LEVIR-CD+_sft.jsonl"),
    )
    process_dataset(
        name="SECOND",
        pairs=second_pairs,
        teacher_prompt_template=SECOND_TEACHER_PROMPT,
        user_instructions=SECOND_USER_INSTRUCTIONS,
        output_file=str(OUTPUT_DIR / "SECOND_sft.jsonl"),
    )

    print("\n" + "=" * 60)
    print("全部完成！输出文件：")
    for name in ["EBD", "LEVIR-CD+", "SECOND"]:
        f = OUTPUT_DIR / f"{name}_sft.jsonl"
        if f.exists():
            count = sum(1 for _ in open(f))
            print(f"  {f}  ({count} 条)")
    print("=" * 60)


if __name__ == "__main__":
    main()
