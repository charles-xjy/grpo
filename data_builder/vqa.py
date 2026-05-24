"""VQA 数据集构造逻辑（单图 → 三阶段流水线）。

Phase 1: 双尺度描述（含图推理）
Phase 2: Q1 开放式问答（纯文本，依赖 Phase 1 描述）
Phase 3: Q2 具体关系/计数问答（纯文本，依赖 Phase 1 描述）

每阶段独立断点续传。
"""

import asyncio
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from . import llm
from .config import VQA_OUTPUT_DIR
from .prompts import ANSWER_TONES, DESC_USER_INSTRUCTIONS, DUAL_SCALE_PROMPT, Q1_GEN_PROMPT, Q2_GEN_PROMPT
from .resume import load_completed, load_desc_map


# ── JSON 解析 ──────────────────────────────────────────────


def _extract_json_array(text: str) -> list | None:
    """从 LLM 响应文本中提取 JSON 数组。"""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r"\[.*\]", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


# ── 人设与口吻池 ──────────────────────────────────────────

QUESTIONERS = [
    ("农民", "问题朴实直接，关注土地、庄稼、水源、天气对农活的影响"),
    ("学生", "问题带着好奇心，求知欲强，用词认真但偶尔稚嫩"),
    ("上班族", "务实简洁，关注交通、建筑、生活便利设施"),
    ("严谨的专家", "使用专业术语，问题精准有条理，注重数据与逻辑"),
    ("新潮的年轻人", "表达新潮随性，偶尔夹带网络用语，语气轻松"),
    ("老年人", "问题简单直接，语速慢，可能带点方言味道，称呼人爱用\"同志\"\"师傅\"等"),
]


def _random_style() -> tuple[str, str]:
    """随机选取提问者身份和回答口吻。"""
    q_label, q_desc = random.choice(QUESTIONERS)
    a_label, a_desc = random.choice(ANSWER_TONES)
    questioner = f"{q_label}（{q_desc}）"
    answer_tone = f"{a_label}（{a_desc}）"
    return questioner, answer_tone


# ── 各阶段 worker ──────────────────────────────────────────


async def _process_desc(img_path: str, f_out, lock: asyncio.Lock, fail_list: list, sem: asyncio.Semaphore, pbar: tqdm):
    try:
        async with sem:
            description = await llm.chat_with_images(DUAL_SCALE_PROMPT, [img_path], max_tokens=2048)
        if not description or len(description.strip()) < 50:
            tqdm.write(f"[跳过] {Path(img_path).name}: 描述过短")
            return
        record = {
            "messages": [
                {"role": "user", "content": f"<image>\n{random.choice(DESC_USER_INSTRUCTIONS)}"},
                {"role": "assistant", "content": description.strip()},
            ],
            "images": [img_path],
        }
        async with lock:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        tqdm.write(f"[错误-desc] {Path(img_path).name}: {e}")
        fail_list.append(img_path)
    finally:
        pbar.update(1)


async def _process_q1(img_path: str, description: str, f_out, lock: asyncio.Lock, fail_list: list, sem: asyncio.Semaphore, pbar: tqdm):
    try:
        questioner, answer_tone = _random_style()
        prompt = Q1_GEN_PROMPT.format(
            num_q=3, questioner=questioner, answer_tone=answer_tone, description=description,
        )
        async with sem:
            q1_raw = await llm.chat_text_only(prompt)
        q1_pairs = _extract_json_array(q1_raw)
        if q1_pairs:
            async with lock:
                for pair in q1_pairs:
                    f_out.write(json.dumps({
                        "messages": [
                            {"role": "user", "content": f"<image>\n{pair['question']}"},
                            {"role": "assistant", "content": pair["answer"]},
                        ],
                        "images": [img_path],
                    }, ensure_ascii=False) + "\n")
                f_out.flush()
        else:
            tqdm.write(f"[Q1解析失败] {Path(img_path).name}: {q1_raw[:200]}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        tqdm.write(f"[错误-Q1] {Path(img_path).name}: {e}")
        fail_list.append(img_path)
    finally:
        pbar.update(1)


async def _process_q2(img_path: str, description: str, f_out, lock: asyncio.Lock, fail_list: list, sem: asyncio.Semaphore, pbar: tqdm):
    try:
        questioner, answer_tone = _random_style()
        prompt = Q2_GEN_PROMPT.format(
            num_q=3, questioner=questioner, answer_tone=answer_tone, description=description,
        )
        async with sem:
            q2_raw = await llm.chat_text_only(prompt)
        q2_pairs = _extract_json_array(q2_raw)
        if q2_pairs:
            async with lock:
                for pair in q2_pairs:
                    thinking = pair.get("thinking", "").strip()
                    answer = pair.get("answer", "").strip()
                    if thinking:
                        full_answer = f"推理过程：\n{thinking}\n\n答案：{answer}"
                    else:
                        full_answer = answer
                    f_out.write(json.dumps({
                        "messages": [
                            {"role": "user", "content": f"<image>\n{pair['question']}"},
                            {"role": "assistant", "content": full_answer},
                        ],
                        "images": [img_path],
                    }, ensure_ascii=False) + "\n")
                f_out.flush()
        else:
            tqdm.write(f"[Q2解析失败] {Path(img_path).name}: {q2_raw[:200]}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        tqdm.write(f"[错误-Q2] {Path(img_path).name}: {e}")
        fail_list.append(img_path)
    finally:
        pbar.update(1)


# ── 主流程 ─────────────────────────────────────────────────


async def run(all_images: list[str], concurrency: int) -> None:
    """三阶段流水线：desc → Q1 → Q2，每阶段独立断点续传。"""
    VQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    desc_file = VQA_OUTPUT_DIR / "dual_scale_desc.jsonl"
    q1_file = VQA_OUTPUT_DIR / "vqa_openended.jsonl"
    q2_file = VQA_OUTPUT_DIR / "vqa_specific.jsonl"

    fail_list: list[str] = []
    sem = asyncio.Semaphore(concurrency)

    # ── Phase 1: 双尺度描述 ──
    done_desc = load_completed(desc_file)
    desc_remaining = [p for p in all_images if p not in done_desc]
    print(f"[Phase 1] 双尺度描述  已完成 {len(done_desc)} / {len(all_images)}, "
          f"待处理 {len(desc_remaining)}")
    if desc_remaining:
        lock_desc = asyncio.Lock()
        with open(desc_file, "a", encoding="utf-8") as f_desc:
            with tqdm(total=len(desc_remaining), desc="Phase 1 双尺度描述") as pbar:
                await asyncio.gather(*[
                    _process_desc(p, f_desc, lock_desc, fail_list, sem, pbar)
                    for p in desc_remaining
                ])

    # ── 加载描述映射 ──
    desc_map = load_desc_map(desc_file)
    print(f"\n描述映射加载: {len(desc_map)} 条")

    # ── Phase 2: Q1 开放式问答 ──
    done_q1 = load_completed(q1_file)
    q1_remaining = [(p, desc_map[p]) for p in all_images if p not in done_q1 and p in desc_map]
    print(f"\n[Phase 2] Q1 开放式问答  已完成 {len(done_q1)} / {len(all_images)}, "
          f"待处理 {len(q1_remaining)}")
    if q1_remaining:
        lock_q1 = asyncio.Lock()
        with open(q1_file, "a", encoding="utf-8") as f_q1:
            with tqdm(total=len(q1_remaining), desc="Phase 2 Q1开放式") as pbar:
                await asyncio.gather(*[
                    _process_q1(p, desc, f_q1, lock_q1, fail_list, sem, pbar)
                    for p, desc in q1_remaining
                ])

    # ── Phase 3: Q2 具体关系/计数 ──
    done_q2 = load_completed(q2_file)
    q2_remaining = [(p, desc_map[p]) for p in all_images if p not in done_q2 and p in desc_map]
    print(f"\n[Phase 3] Q2 具体关系/计数  已完成 {len(done_q2)} / {len(all_images)}, "
          f"待处理 {len(q2_remaining)}")
    if q2_remaining:
        lock_q2 = asyncio.Lock()
        with open(q2_file, "a", encoding="utf-8") as f_q2:
            with tqdm(total=len(q2_remaining), desc="Phase 3 Q2具体关系") as pbar:
                await asyncio.gather(*[
                    _process_q2(p, desc, f_q2, lock_q2, fail_list, sem, pbar)
                    for p, desc in q2_remaining
                ])

    # ── 统计 ──
    print("\n" + "=" * 60)
    print("完成！输出文件：")
    for label, f in [("双尺度描述", desc_file), ("Q1-开放式VQA", q1_file), ("Q2-具体关系VQA", q2_file)]:
        if f.exists():
            count = sum(1 for _ in open(f))
            print(f"  {label}: {f}  ({count} 条)")
    if fail_list:
        print(f"\n失败 {len(fail_list)} 条（下次运行自动重试）")
    print("=" * 60)
