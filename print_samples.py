#!/usr/bin/env python3
"""打印所有 SFT 和 VQA 数据文件的前10条记录。"""
import json
import os

from data_builder.config import SFT_OUTPUT_DIR, VQA_OUTPUT_DIR

FILES = {
    "SFT": [
        str(SFT_OUTPUT_DIR / "EBD_sft.jsonl"),
        str(SFT_OUTPUT_DIR / "LEVIR-CD+_sft.jsonl"),
        str(SFT_OUTPUT_DIR / "SECOND_sft.jsonl"),
    ],
    "VQA": [
        str(VQA_OUTPUT_DIR / "dual_scale_desc.jsonl"),
        str(VQA_OUTPUT_DIR / "vqa_openended.jsonl"),
        str(VQA_OUTPUT_DIR / "vqa_specific.jsonl"),
    ],
}

N = 10

for category, paths in FILES.items():
    for path in paths:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"\n{'=' * 70}\n[{category}] {name}  — 文件不存在，跳过\n{'=' * 70}")
            continue
        total = sum(1 for _ in open(path))
        print(f"\n{'=' * 70}\n[{category}] {name}  (共 {total} 条，展示前 {N} 条)\n{'=' * 70}")
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= N:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                images = rec.get("images", [])
                print(f"\n--- 第 {i + 1} 条 ---")
                print(f"  images ({len(images)}):")
                for img in images:
                    print(f"    {os.path.basename(img)}")
                for msg in rec["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    # 截断过长内容
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"  [{role}]: {content}")
