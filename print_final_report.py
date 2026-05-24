#!/usr/bin/env python3
"""最终汇总报告：对比 [微调前] vs [微调后] vs [老师天花板]。"""

import json
from pathlib import Path
from data_builder.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "evaluation"
BASELINE_DIR = EVAL_DIR / "baseline"

def get_stats(dir_path):
    cot_file = dir_path / "eval_cot_results.jsonl"
    vqa_file = dir_path / "eval_vqa_results.jsonl"
    
    res = {"cot_s": 0, "cot_t": 0, "q1_s": 0, "q1_t": 0, "q2_s": 0, "q2_t": 0}
    
    if cot_file.exists():
        lines = [json.loads(l) for l in open(cot_file)]
        res["cot_s"] = sum(l.get("score", 0) for l in lines) / len(lines)
        res["cot_t"] = sum(l.get("teacher_self_score", 0) for l in lines) / len(lines)
        
    if vqa_file.exists():
        lines = [json.loads(l) for l in open(vqa_file)]
        q1_s, q1_t, q2_s, q2_t = [], [], [], []
        for l in lines:
            q1_s.extend([i.get("score", 0) for i in l["q1_eval"]])
            q1_t.extend([i.get("teacher_exam_score", 0) for i in l["q1_eval"]])
            q2_s.extend([i.get("score", 0) for i in l["q2_eval"]])
            q2_t.extend([i.get("teacher_exam_score", 0) for i in l["q2_eval"]])
        res["q1_s"] = sum(q1_s)/len(q1_s) if q1_s else 0
        res["q1_t"] = sum(q1_t)/len(q1_t) if q1_t else 0
        res["q2_s"] = sum(q2_s)/len(q2_s) if q2_s else 0
        res["q2_t"] = sum(q2_t)/len(q2_t) if q2_t else 0
    return res

def main():
    print("\n正在汇总测评数据...")
    after = get_stats(EVAL_DIR)
    before = get_stats(BASELINE_DIR)
    
    print(f"\n{'='*75}")
    print(f"  遥感大模型能力演进报告 (综合对比)")
    print(f"{'='*75}")
    print(f"  任务类型            微调前      微调后      老师(上限)    提升幅度")
    print(f"  {'─'*71}")
    
    def row(label, b, a, t):
        gain = ((a - b) / b * 100) if b > 0 else 0
        print(f"  {label:<18s} {b:>6.2f}  ──→  {a:>6.2f}      {t:>6.2f}      {gain:>+6.1f}%")

    row("1. Change Det (CoT)", before["cot_s"], after["cot_s"], after["cot_t"])
    row("2. VQA-Open (Q1)", before["q1_s"], after["q1_s"], after["q1_t"])
    row("3. VQA-Reason (Q2)", before["q2_s"], after["q2_s"], after["q2_t"])
    
    print(f"{'='*75}")
    print(f"  结论：微调后模型在 Q2 任务上相比 Baseline 提升了 {((after['q2_s']-before['q2_s'])/before['q2_s']*100 if before['q2_s']>0 else 0):.1f}%。")

if __name__ == "__main__":
    main()
