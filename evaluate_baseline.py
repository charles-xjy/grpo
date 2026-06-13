#!/usr/bin/env python3
"""基准评估脚本：使用微调前模型 (8003 端口) 答题。

复用已有的测试集 (evaluation/eval_*.jsonl)，仅进行答题，不重新出题。
结果将保存在 evaluation/baseline/ 目录下。
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from data_builder import llm
from data_builder.config import PROJECT_ROOT
from data_builder.prompts import SECOND_TEACHER_PROMPT

SRC_EVAL_DIR = PROJECT_ROOT / "evaluation"
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "baseline"

async def _process_cot_baseline(baseline_client, record, f_out, lock, sem, pbar):
    """Task 1: 微调前模型生成变化检测报告"""
    try:
        pre_img = record["pre_img"]
        post_img = record["post_img"]
        
        # 直接提取原来的 User Content，不使用系统提示词，不硬编码
        prompt = record.get("eval_prompt")
        if not prompt:
            # 兜底逻辑：如果源文件没存，用最简单的格式
            prompt = "<image>\n<image>\n请对比分析这两幅影像的地表覆盖变化。"
        
        async with sem:
            baseline_report = await baseline_client.chat_with_images(prompt, [pre_img, post_img])
            
        new_record = {
            "pre_img": pre_img,
            "post_img": post_img,
            "teacher_report": record["teacher_report"], # 保留参考答案
            "student_report": baseline_report,          # 替换为 baseline 的答案
            "score": None
        }
        
        async with lock:
            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            f_out.flush()
    except Exception as e:
        tqdm.write(f"[Error-CoT] {Path(record['pre_img']).name}: {e}")
    finally:
        pbar.update(1)

async def _process_vqa_baseline(baseline_client, record, f_out, lock, sem, pbar):
    """Task 2: 微调前模型回答 VQA 问题"""
    try:
        img_path = record["img_path"]
        
        q1_results = []
        q2_results = []
        
        async with sem:
            # 答 Q1
            for item in record["q1_eval"]:
                q = item["question"]
                b_ans = await baseline_client.chat_with_images(f"<image>\n{q}", [img_path])
                q1_results.append({
                    "question": q,
                    "teacher_answer": item["teacher_answer"],
                    "student_answer": b_ans, # 替换为 baseline 的答案
                    "score": None
                })

            # 答 Q2
            for item in record["q2_eval"]:
                q = item["question"]
                # 仅提供原始问题，不加辅助引导语
                b_raw = await baseline_client.chat_with_images(f"<image>\n{q}", [img_path])
                q2_results.append({
                    "question": q,
                    "teacher_thinking": item.get("teacher_thinking"),
                    "teacher_answer": item["teacher_answer"],
                    "student_full_response": b_raw, # 替换为 baseline 的答案
                    "score": None
                })

        new_record = {
            "img_path": img_path,
            "teacher_description": record["teacher_description"],
            "q1_eval": q1_results,
            "q2_eval": q2_results,
        }
        
        async with lock:
            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            f_out.flush()
    except Exception as e:
        tqdm.write(f"[Error-VQA] {Path(record['img_path']).name}: {e}")
    finally:
        pbar.update(1)

def load_completed_images(file_path: Path, key: str = "img_path") -> set[str]:
    completed = set()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if key in data: completed.add(data[key])
                    elif "pre_img" in data: completed.add(data["pre_img"])
                except: continue
    return completed

async def main():
    parser = argparse.ArgumentParser(description="Baseline Model (Port 8003) Evaluation")
    parser.add_argument("--baseline-url", default="http://localhost:8003", help="Baseline vLLM URL")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency level")
    args = parser.parse_args()

    # 1. 检查数据源是否存在
    src_cot_file = SRC_EVAL_DIR / "eval_cot_results.jsonl"
    src_vqa_file = SRC_EVAL_DIR / "eval_vqa_results.jsonl"
    if not src_cot_file.exists() or not src_vqa_file.exists():
        print(f"FAIL: Source evaluation files not found in {SRC_EVAL_DIR}. Please run evaluate_model.py first.")
        sys.exit(1)

    # 2. 初始化基准模型
    b_model = llm.fetch_available_model(args.baseline_url)
    if not b_model:
        print(f"FAIL: Baseline model not found at {args.baseline_url}")
        sys.exit(1)
    
    print(f"Baseline Model: {b_model} at {args.baseline_url}")
    baseline_client = llm.LLMClient(args.baseline_url, b_model)
    sem = asyncio.Semaphore(args.concurrency)
    
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 处理 CoT
    tgt_cot_file = BASELINE_OUTPUT_DIR / "eval_cot_results.jsonl"
    done_cot = load_completed_images(tgt_cot_file, "pre_img")
    
    cot_records = [json.loads(l) for l in open(src_cot_file, encoding="utf-8")]
    cot_eval_set = [r for r in cot_records if r["pre_img"] not in done_cot]
    
    print(f"\n[Task 1] Baseline CoT: Total {len(cot_records)}, Done {len(done_cot)}, Remaining {len(cot_eval_set)}")
    if cot_eval_set:
        lock = asyncio.Lock()
        with open(tgt_cot_file, "a", encoding="utf-8") as f:
            with tqdm(total=len(cot_eval_set), desc="Baseline CoT") as pbar:
                await asyncio.gather(*[
                    _process_cot_baseline(baseline_client, r, f, lock, sem, pbar) 
                    for r in cot_eval_set
                ])

    # 4. 处理 VQA
    tgt_vqa_file = BASELINE_OUTPUT_DIR / "eval_vqa_results.jsonl"
    done_vqa = load_completed_images(tgt_vqa_file, "img_path")
    
    vqa_records = [json.loads(l) for l in open(src_vqa_file, encoding="utf-8")]
    vqa_eval_set = [r for r in vqa_records if r["img_path"] not in done_vqa]
    
    print(f"\n[Task 2] Baseline VQA: Total {len(vqa_records)}, Done {len(done_vqa)}, Remaining {len(vqa_eval_set)}")
    if vqa_eval_set:
        lock = asyncio.Lock()
        with open(tgt_vqa_file, "a", encoding="utf-8") as f:
            with tqdm(total=len(vqa_eval_set), desc="Baseline VQA") as pbar:
                await asyncio.gather(*[
                    _process_vqa_baseline(baseline_client, r, f, lock, sem, pbar)
                    for r in vqa_eval_set
                ])

    print(f"\nBaseline evaluation data generated in: {BASELINE_OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
