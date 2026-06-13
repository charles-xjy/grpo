"""SFT 数据集构造逻辑（图像对 → 分析报告）。

用 asyncio + Semaphore 替换了旧的 ThreadPoolExecutor，
保持与原有行为一致：并行请求、实时追加写入、断点续传、失败跳过。
"""

import asyncio
import json
import random
from pathlib import Path

from tqdm import tqdm

from . import llm
from .config import SFT_OUTPUT_DIR
from .prompts import ANSWER_TONES, SFT_PERSONA_PREFIXES
from .resume import load_completed


def _random_tone() -> str:
    label, desc = random.choice(ANSWER_TONES)
    return f"{label}（{desc}）"


def _random_instruction(tasks: list[str]) -> str:
    """随机组装人设前缀 + 数据集任务 → 多样化的 user 提问。"""
    persona_prefixes = random.choice(list(SFT_PERSONA_PREFIXES.values()))
    prefix = random.choice(persona_prefixes)
    task = random.choice(tasks)
    return f"{prefix}{task}"


async def _process_single(task: dict, teacher_prompt: str, tasks: list[str], sem: asyncio.Semaphore) -> dict | None:
    """处理一对图像，生成一条 SFT 记录。失败返回 None。"""
    pre_path = str(task["pre"])
    post_path = str(task["post"])
    prompt = teacher_prompt.format(answer_tone=_random_tone())

    async with sem:
        try:
            answer = await llm.chat_with_images(
                prompt=prompt,
                image_paths=[pre_path, post_path],
                temperature=0.7,
                max_tokens=2048,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    if not answer:
        return None

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n<image>\n{_random_instruction(tasks)}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(Path(pre_path).resolve()), str(Path(post_path).resolve())],
    }


async def process_dataset(
    name: str,
    pairs: list[dict],
    teacher_prompt: str,
    tasks: list[str],
    output_file: str,
    concurrency: int,
) -> None:
    """处理一个数据集的全部图像对（异步并行，断点续传）。"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传
    done = load_completed(output_path)
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

    fail_list: list[str] = []
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def _worker(task: dict, pbar: tqdm) -> None:
        nonlocal fail_list
        record = await _process_single(task, teacher_prompt, tasks, sem)
        if record is not None:
            async with write_lock:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
        else:
            stem = Path(task["pre"]).stem
            tqdm.write(f"\n[FAIL {name}] {stem}: retries exhausted")
            fail_list.append(str(task["pre"]))
        pbar.update(1)

    with open(output_path, "a", encoding="utf-8") as f_out:
        with tqdm(total=len(remaining), desc=name) as pbar:
            futures = [asyncio.create_task(_worker(t, pbar)) for t in remaining]
            try:
                await asyncio.gather(*futures)
            except KeyboardInterrupt:
                remaining_count = sum(1 for t in futures if not t.done())
                print(f"\n用户中断，取消剩余 {remaining_count} 个任务...")
                for t in futures:
                    t.cancel()
                if fail_list:
                    print(f"本次失败 {len(fail_list)} 条，下次运行会自动重试。")
                return

    success = len(remaining) - len(fail_list)
    if fail_list:
        print(f"\n{name} 处理完成，成功 {success} 条，失败 {len(fail_list)} 条（下次运行会自动重试）：")
        for fp in fail_list[:10]:
            print(f"  - {fp}")
        if len(fail_list) > 10:
            print(f"  ... 及其他 {len(fail_list) - 10} 条")
    else:
        print(f"\n{name} 处理完成，成功 {success} 条")


async def run_all(
    ebd_pairs: list[dict],
    levir_pairs: list[dict],
    second_pairs: list[dict],
    concurrency: int,
) -> None:
    """运行全部三个数据集的 SFT 数据构造。"""
    from .prompts import (
        EBD_SFT_TASKS, EBD_TEACHER_PROMPT,
        LEVIR_SFT_TASKS, LEVIR_TEACHER_PROMPT,
        SECOND_SFT_TASKS, SECOND_TEACHER_PROMPT,
    )

    await process_dataset(
        name="EBD", pairs=ebd_pairs,
        teacher_prompt=EBD_TEACHER_PROMPT, tasks=EBD_SFT_TASKS,
        output_file=str(SFT_OUTPUT_DIR / "EBD_sft.jsonl"), concurrency=concurrency,
    )
    await process_dataset(
        name="LEVIR-CD+", pairs=levir_pairs,
        teacher_prompt=LEVIR_TEACHER_PROMPT, tasks=LEVIR_SFT_TASKS,
        output_file=str(SFT_OUTPUT_DIR / "LEVIR-CD+_sft.jsonl"), concurrency=concurrency,
    )
    await process_dataset(
        name="SECOND", pairs=second_pairs,
        teacher_prompt=SECOND_TEACHER_PROMPT, tasks=SECOND_SFT_TASKS,
        output_file=str(SFT_OUTPUT_DIR / "SECOND_sft.jsonl"), concurrency=concurrency,
    )

    print("\n" + "=" * 60)
    print("全部完成！输出文件：")
    for name in ["EBD", "LEVIR-CD+", "SECOND"]:
        f = SFT_OUTPUT_DIR / f"{name}_sft.jsonl"
        if f.exists():
            count = sum(1 for _ in open(f))
            print(f"  {f}  ({count} 条)")
    print("=" * 60)
