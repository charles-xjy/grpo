#!/usr/bin/env python3
"""统计 SFT 和 VQA 数据文件的记录数。"""
import json
from pathlib import Path

from data_builder.config import SFT_OUTPUT_DIR, VQA_OUTPUT_DIR

files = {
    "SFT": [
        SFT_OUTPUT_DIR / "EBD_sft.jsonl",
        SFT_OUTPUT_DIR / "LEVIR-CD+_sft.jsonl",
        SFT_OUTPUT_DIR / "SECOND_sft.jsonl",
    ],
    "VQA": [
        VQA_OUTPUT_DIR / "dual_scale_desc.jsonl",
        VQA_OUTPUT_DIR / "vqa_openended.jsonl",
        VQA_OUTPUT_DIR / "vqa_specific.jsonl",
    ],
}

grand_total = 0
for category, paths in files.items():
    print(f"\n{'=' * 55}")
    print(f"  {category}")
    print(f"{'=' * 55}")
    cat_total = 0
    for p in paths:
        if p.exists():
            count = sum(1 for _ in open(p))
            cat_total += count
            print(f"  {p.name:<35s} {count:>6d} 条")
        else:
            print(f"  {p.name:<35s}  (不存在)")
    print(f"  {'─' * 45}")
    print(f"  {category} 小计: {cat_total} 条")
    grand_total += cat_total

print(f"\n{'=' * 55}")
print(f"  总计: {grand_total} 条")
print(f"{'=' * 55}")
