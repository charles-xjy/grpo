#!/usr/bin/env python3
"""评分脚本：老师考学生，也考自己。结果写回原 JSONL 文件。

新增字段：
- Task 1: teacher_self_score
- Task 2: teacher_q1_score, teacher_q2_score, teacher_exam_answers
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

from data_builder import llm
from data_builder.config import PROJECT_ROOT

EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation"

# ── 打分 Prompt 模板 ──────────────────────────────────────

REPORT_SCORE_PROMPT = """# Role
你是一位权威的遥感图像解译专家。请对这份“遥感变化检测报告”进行质量打分。

# Context
报告内容：{report}

# Scoring Criteria (0-10分)
- **准确性 (4分)**：是否准确识别了地表变化？是否存在逻辑矛盾？
- **细节程度 (3分)**：是否锁定了具体的空间位置？
- **专业性 (3分)**：术语使用是否规范，结构是否清晰。

# Output Format
最后以 "Score: [分数]" 结尾。
"""

QA_SCORE_PROMPT = """# Role
请作为裁判对 VQA 问答进行打分。

# Context
- 问题：{question}
- 参考答案：{reference_answer}
- 待评答案：{candidate_answer}

# Scoring Criteria (0-10分)
- 10分：回答完美，涵盖了参考答案的所有要点。
- 0-3分：回答错误，或者与参考答案严重冲突。

# Output Format
最后以 "Score: [分数]" 结尾。
"""

def _extract_score(text: str) -> float:
    match = re.search(r"Score:\s*\[?(\d+(?:\.\d+)?)\]?", text, re.IGNORECASE)
    return min(10.0, max(0.0, float(match.group(1)))) if match else 0.0

# ── 评分 Worker ──────────────────────────────────────────

def safe_avg(scores: list) -> float:
    """安全计算平均分，处理空列表。"""
    return sum(scores) / len(scores) if scores else 0.0

async def process_cot(teacher, record, sem):
    # 彻底的断点续传检查：必须两者都有分才跳过
    if record.get("score") is not None and record.get("teacher_self_score") is not None:
        return record["teacher_self_score"], record["score"]

    async with sem:
        # 评老师自己当初生成的报告
        if record.get("teacher_self_score") is None:
            t_judge = await teacher.chat_text_only(REPORT_SCORE_PROMPT.format(report=record["teacher_report"]))
            record["teacher_self_score"] = _extract_score(t_judge)
        
        # 评学生生成的报告
        if record.get("score") is None:
            s_judge = await teacher.chat_text_only(REPORT_SCORE_PROMPT.format(report=record["student_report"]))
            record["score"] = _extract_score(s_judge)
    return record["teacher_self_score"], record["score"]

async def score_single_qa(teacher, img_path, item, is_q2, sem):
    """处理单个问答对：老师答题 + 裁判打分。"""
    if item.get("score") is not None and item.get("teacher_exam_score") is not None:
        return item["teacher_exam_score"], item["score"]

    async with sem:
        # 1. 老师现场闭卷考试
        suffix = "\n请先推理再回答。" if is_q2 else ""
        t_ans = await teacher.chat_with_images(f"<image>\n{item['question']}{suffix}", [img_path])
        ref = item["teacher_answer"]
        
        # 2. 裁判打分 (并行两个打分请求)
        t_task = teacher.chat_text_only(QA_SCORE_PROMPT.format(
            question=item['question'], reference_answer=ref, candidate_answer=t_ans))
        s_task = teacher.chat_text_only(QA_SCORE_PROMPT.format(
            question=item['question'], reference_answer=ref, candidate_answer=item["student_answer"] if not is_q2 else item["student_full_response"]))
        
        t_score_res, s_score_res = await asyncio.gather(t_task, s_task)
        
        item["teacher_exam_score"] = _extract_score(t_score_res)
        item["score"] = _extract_score(s_score_res)
        
    return item["teacher_exam_score"], item["score"]

async def process_vqa(teacher, record, sem):
    img_path = record["img_path"]
    
    # 将 Q1 和 Q2 的所有子任务全部并行化
    tasks = []
    for item in record["q1_eval"]:
        tasks.append(score_single_qa(teacher, img_path, item, False, sem))
    for item in record["q2_eval"]:
        tasks.append(score_single_qa(teacher, img_path, item, True, sem))
    
    if not tasks:
        return 0.0, 0.0, 0.0, 0.0
        
    await asyncio.gather(*tasks)

    t_q1_scores = [i.get("teacher_exam_score", 0) for i in record["q1_eval"]]
    s_q1_scores = [i.get("score", 0) for i in record["q1_eval"]]
    t_q2_scores = [i.get("teacher_exam_score", 0) for i in record["q2_eval"]]
    s_q2_scores = [i.get("score", 0) for i in record["q2_eval"]]
    
    return safe_avg(t_q1_scores), safe_avg(s_q1_scores), safe_avg(t_q2_scores), safe_avg(s_q2_scores)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-url", default="http://localhost:8002")
    parser.add_argument("--input-dir", default="evaluation", help="Directory containing eval_*.jsonl files")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    if not input_dir.exists():
        print(f"FAIL: Input directory {input_dir} does not exist.")
        sys.exit(1)

    t_model = llm.fetch_available_model(args.teacher_url)
    teacher = llm.LLMClient(args.teacher_url, t_model)
    sem = asyncio.Semaphore(args.concurrency)

    # CoT 评分
    cot_file = input_dir / "eval_cot_results.jsonl"
    cot_stats = [0.0, 0.0]
    if cot_file.exists():
        print(f"\n[Scoring CoT from {args.input_dir}]...")
        recs = [json.loads(l) for l in open(cot_file, encoding="utf-8")]
        tasks = [process_cot(teacher, r, sem) for r in recs]
        res = [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        cot_stats = [sum(x)/len(recs) for x in zip(*res)]
        with open(cot_file, "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # VQA 评分
    vqa_file = input_dir / "eval_vqa_results.jsonl"
    vqa_stats = [0.0] * 4
    if vqa_file.exists():
        print(f"\n[Scoring VQA from {args.input_dir}]...")
        recs = [json.loads(l) for l in open(vqa_file, encoding="utf-8")]
        tasks = [process_vqa(teacher, r, sem) for r in recs]
        res = [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        vqa_stats = [sum(x)/len(recs) for x in zip(*res)]
        with open(vqa_file, "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}\n  能力对比报告 (学生 vs 老师)\n{'='*50}")
    print(f"  CoT:   学生 {cot_stats[1]:>5.2f} | 老师 {cot_stats[0]:>5.2f}")
    print(f"  VQA Q1: 学生 {vqa_stats[1]:>5.2f} | 老师 {vqa_stats[0]:>5.2f}")
    print(f"  VQA Q2: 学生 {vqa_stats[3]:>5.2f} | 老师 {vqa_stats[2]:>5.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
