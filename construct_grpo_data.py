#!/usr/bin/env python3
"""构造 GRPO 训练数据集 (MS-Swift 格式)。

采样规模：
- EBD: 300 对
- LEVIR-CD+: 500 对
- SECOND: 1500 对

Swift 格式要求：
- messages: [{"role": "user", "content": "<image>\n<image>\n..."}]
- images: [abs_path1, abs_path2]
"""

import json
import random
from pathlib import Path
from collections import defaultdict

from data_builder.sources import build_ebd_pairs, build_levir_pairs, build_second_pairs
from data_builder.config import EBD_PER_EVENT, LEVIR_TAKE, SECOND_TAKE, PROJECT_ROOT
from data_builder.prompts import (
    SFT_PERSONA_PREFIXES,
    EBD_SFT_TASKS,
    LEVIR_SFT_TASKS,
    SECOND_SFT_TASKS,
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "grpo_swift.jsonl"

def get_random_prompt(tasks):
    """合成随机人设 + 随机指令。"""
    persona_list = random.choice(list(SFT_PERSONA_PREFIXES.values()))
    prefix = random.choice(persona_list)
    task = random.choice(tasks)
    return f"{prefix}\n{task}"

def make_swift_item(pair, tasks):
    """按照 Swift 要求封装单条数据。"""
    prompt_text = f"<image>\n<image>\n{get_random_prompt(tasks)}"
    return {
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "images": [
            str(pair["pre"].resolve()),
            str(pair["post"].resolve())
        ]
    }

def main():
    grpo_data = []

    # 1. EBD 采样 (每种灾害 300 对)
    ebd_all = build_ebd_pairs()
    ebd_by_event = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    
    ebd_count = 0
    for event, pairs in ebd_by_event.items():
        # 排除 SFT 占用的前 N 对
        remaining = pairs[EBD_PER_EVENT:]
        random.shuffle(remaining)
        sampled = remaining[:300]
        ebd_count += len(sampled)
        for p in sampled:
            grpo_data.append(make_swift_item(p, EBD_SFT_TASKS))
    print(f"  EBD 采样完成: 每种灾害取 300，共计 {ebd_count} 条")

    # 2. LEVIR-CD+ 采样 (500)
    levir_all = build_levir_pairs()
    levir_remaining = levir_all[LEVIR_TAKE:]
    random.shuffle(levir_remaining)
    levir_sample = levir_remaining[:500]
    for p in levir_sample:
        grpo_data.append(make_swift_item(p, LEVIR_SFT_TASKS))

    # 3. SECOND 采样 (1500)
    second_all = build_second_pairs()
    second_remaining = second_all[700:] 
    random.shuffle(second_remaining)
    second_sample = second_remaining[:1500]
    for p in second_sample:
        grpo_data.append(make_swift_item(p, SECOND_SFT_TASKS))

    # 随机打乱
    random.shuffle(grpo_data)

    # 写入文件
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSwift 格式 GRPO 数据集构造完成：")
    print(f"  EBD:    {ebd_count} 条")
    print(f"  LEVIR:  {len(levir_sample)} 条")
    print(f"  SECOND: {len(second_sample)} 条")
    print(f"  总计:   {len(grpo_data)} 条")
    print(f"  保存位置: {OUTPUT_FILE}")
    print("\n请将以下内容添加到 data/dataset_info.json 中：")
    print(json.dumps({
        "remote_sensing_grpo": {
            "file_name": "grpo_swift.jsonl",
            "columns": {
                "prompt": "messages",
                "images": "images"
            }
        }
    }, indent=2))

if __name__ == "__main__":
    main()
