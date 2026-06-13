"""入口脚本：构造 VQA 数据集（单图 → 双尺度描述 + 开放式问答 + 具体关系问答）。"""

import argparse
import asyncio
import os
import random
import sys

from data_builder import llm
from data_builder.sources import collect_vqa_candidates, print_vqa_summary
from data_builder.vqa import run


def main():
    parser = argparse.ArgumentParser(description="Construct VQA data from teacher model")
    parser.add_argument("--model", default=None,
                        help="Model name (default: $VLLM_MODEL or auto-detect)")
    parser.add_argument("--base-url", default="http://localhost:8002",
                        help="vLLM server base URL")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of parallel requests")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries per API call")
    args = parser.parse_args()

    model_name = args.model or os.getenv("VLLM_MODEL") or llm.fetch_available_model(args.base_url)
    if not model_name:
        print("FAIL: unable to determine model. Pass --model or set $VLLM_MODEL.")
        sys.exit(1)
    print(f"Using model: {model_name}")

    llm.init(args.base_url, model_name, args.retries)

    print("计算 CoT 数据占用的 post 侧图像...")
    per_dataset = collect_vqa_candidates()
    print_vqa_summary(per_dataset)

    all_images = []
    for imgs in per_dataset.values():
        all_images.extend(imgs)
    random.shuffle(all_images)
    print(f"  总计: {len(all_images)} 张 post 图像\n")

    asyncio.run(run(all_images, args.concurrency))


if __name__ == "__main__":
    main()
