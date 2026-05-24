#!/usr/bin/env python3
"""评估脚本：老师出题 (8002 端口)，学生答题 (8001 端口)。

1. Task 1 (CoT): 老师和学生各自对 SECOND 影像对生成变化检测报告。
2. Task 2 (VQA): 老师看图出题 (Q1/Q2) 并给参考答案，学生闭卷答题。

特点：
- 并行解耦：老师和学生使用独立的信号量（默认各为 4），充分利用多显卡资源。
- 流水线作业：老师出完题后立即释放资源处理下一张，不等待学生答题。
- 断点续传：自动跳过已处理的图片。
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from data_builder import llm
from data_builder.config import SECOND_TAKE, PROJECT_ROOT
from data_builder.sources import build_second_pairs
from data_builder.prompts import (
    SECOND_TEACHER_PROMPT,
    DUAL_SCALE_PROMPT,
    Q1_GEN_PROMPT,
    Q2_GEN_PROMPT,
    SFT_PERSONA_PREFIXES,
    SECOND_SFT_TASKS
)
from data_builder.vqa import _extract_json_array, _random_style

EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation"

def get_random_prompt(tasks):
    """合成随机人设 + 随机指令（对齐训练格式）。"""
    import random
    persona_list = random.choice(list(SFT_PERSONA_PREFIXES.values()))
    prefix = random.choice(persona_list)
    task = random.choice(tasks)
    return f"{prefix}\n{task}"

async def _process_cot_eval(pair, teacher, student, f_out, lock, t_sem, s_sem, pbar):
    """Task 1: Change Detection CoT - Both generate reports."""
    try:
        pre_img = str(pair["pre"].resolve())
        post_img = str(pair["post"].resolve())
        
        # 现场生成一个随机 Prompt 并保存，确保对齐训练分布
        raw_prompt = get_random_prompt(SECOND_SFT_TASKS)
        prompt = f"<image>\n<image>\n{raw_prompt}"
        
        async with t_sem:
            teacher_report = await teacher.chat_with_images(prompt, [pre_img, post_img])
        
        async with s_sem:
            student_report = await student.chat_with_images(prompt, [pre_img, post_img])
            
        record = {
            "pre_img": pre_img,
            "post_img": post_img,
            "eval_prompt": prompt,  # 记录这个 Prompt，方便 baseline 提取
            "teacher_report": teacher_report,
            "student_report": student_report,
            "score": None
        }
        
        async with lock:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
    except Exception as e:
        tqdm.write(f"[Error-CoT] {pair['pre'].name}: {e}")
    finally:
        pbar.update(1)


async def _process_vqa_eval(pair, teacher, student, f_out, lock, t_sem, s_sem, pbar):
    """Task 2: VQA Evaluation - Teacher creates paper, Student takes exam."""
    try:
        img_path = str(pair["post"].resolve())
        
        # --- 老师阶段 (Teacher) ---
        async with t_sem:
            # 1. 老师生成描述
            desc = await teacher.chat_with_images(DUAL_SCALE_PROMPT, [img_path])
            if not desc: return

            questioner, answer_tone = _random_style()
            
            # 2. 老师出 Q1 题
            q1_prompt = Q1_GEN_PROMPT.format(num_q=2, questioner=questioner, answer_tone=answer_tone, description=desc)
            q1_raw = await teacher.chat_text_only(q1_prompt)
            q1_teacher_items = _extract_json_array(q1_raw) or []
            
            # 3. 老师出 Q2 题
            q2_prompt = Q2_GEN_PROMPT.format(num_q=2, questioner=questioner, answer_tone=answer_tone, description=desc)
            q2_raw = await teacher.chat_text_only(q2_prompt)
            q2_teacher_items = _extract_json_array(q2_raw) or []

        # --- 学生阶段 (Student) ---
        async with s_sem:
            q1_results = []
            for item in q1_teacher_items:
                q = item.get("question")
                # 学生只看图和题
                s_ans = await student.chat_with_images(f"<image>\n{q}", [img_path])
                q1_results.append({
                    "question": q,
                    "teacher_answer": item.get("answer"),
                    "student_answer": s_ans,
                    "score": None
                })

            q2_results = []
            for item in q2_teacher_items:
                q = item.get("question")
                # 学生挑战 Q2，仅提供原始问题
                s_raw = await student.chat_with_images(f"<image>\n{q}", [img_path])
                q2_results.append({
                    "question": q,
                    "teacher_thinking": item.get("thinking"),
                    "teacher_answer": item.get("answer"),
                    "student_full_response": s_raw,
                    "score": None
                })

        record = {
            "img_path": img_path,
            "teacher_description": desc,
            "q1_eval": q1_results,
            "q2_eval": q2_results,
            "score_desc_quality": None
        }
        
        async with lock:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
    except Exception as e:
        tqdm.write(f"[Error-VQA] {pair['post'].name}: {e}")
    finally:
        pbar.update(1)


def load_completed_images(file_path: Path, key: str = "img_path") -> set[str]:
    """读取已完成的记录，提取图片路径。"""
    completed = set()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if key in data:
                        completed.add(data[key])
                    elif "pre_img" in data: # 特殊处理 CoT 的 key
                        completed.add(data["pre_img"])
                except:
                    continue
    return completed


async def main():
    parser = argparse.ArgumentParser(description="Teacher-Student Evaluation on SECOND dataset")
    parser.add_argument("--teacher-url", default="http://localhost:8002", help="Teacher vLLM URL")
    parser.add_argument("--student-url", default="http://localhost:8001", help="Student vLLM URL")
    parser.add_argument("--t-concurrency", type=int, default=8, help="Teacher concurrency")
    parser.add_argument("--s-concurrency", type=int, default=8, help="Student concurrency")
    args = parser.parse_args()

    # 自动探测模型名称
    t_model = llm.fetch_available_model(args.teacher_url)
    s_model = llm.fetch_available_model(args.student_url)
    if not t_model or not s_model:
        print(f"FAIL: Teacher ({t_model}) or Student ({s_model}) model not found.")
        sys.exit(1)

    print(f"Teacher Model: {t_model} at {args.teacher_url}")
    print(f"Student Model: {s_model} at {args.student_url}")

    teacher_client = llm.LLMClient(args.teacher_url, t_model)
    student_client = llm.LLMClient(args.student_url, s_model)

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_second = build_second_pairs()
    remaining = all_second[SECOND_TAKE:]
    
    cot_eval_set_all = remaining[:100]
    vqa_eval_set_all = remaining[100:200]
    
    t_sem = asyncio.Semaphore(args.t_concurrency)
    s_sem = asyncio.Semaphore(args.s_concurrency)
    
    # Task 1: CoT
    if cot_eval_set_all:
        cot_file = EVAL_OUTPUT_DIR / "eval_cot_results.jsonl"
        done_cot = load_completed_images(cot_file, "pre_img")
        cot_eval_set = [p for p in cot_eval_set_all if str(p["pre"].resolve()) not in done_cot]
        
        print(f"\n[Task 1] CoT Evaluation: Total 100, Done {len(done_cot)}, Remaining {len(cot_eval_set)}")
        if cot_eval_set:
            lock = asyncio.Lock()
            with open(cot_file, "a", encoding="utf-8") as f:
                with tqdm(total=len(cot_eval_set), desc="CoT Eval") as pbar:
                    await asyncio.gather(*[
                        _process_cot_eval(p, teacher_client, student_client, f, lock, t_sem, s_sem, pbar) 
                        for p in cot_eval_set
                    ])
    
    # Task 2: VQA
    if vqa_eval_set_all:
        vqa_file = EVAL_OUTPUT_DIR / "eval_vqa_results.jsonl"
        done_vqa = load_completed_images(vqa_file, "img_path")
        vqa_eval_set = [p for p in vqa_eval_set_all if str(p["post"].resolve()) not in done_vqa]
        
        print(f"\n[Task 2] VQA Evaluation: Total 100, Done {len(done_vqa)}, Remaining {len(vqa_eval_set)}")
        if vqa_eval_set:
            lock = asyncio.Lock()
            with open(vqa_file, "a", encoding="utf-8") as f:
                with tqdm(total=len(vqa_eval_set), desc="VQA Eval") as pbar:
                    await asyncio.gather(*[
                        _process_vqa_eval(p, teacher_client, student_client, f, lock, t_sem, s_sem, pbar)
                        for p in vqa_eval_set
                    ])

    print(f"\nEvaluation data generated in: {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
