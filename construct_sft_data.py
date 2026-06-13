"""入口脚本：构造 SFT 变化检测数据集（图像对 → 分析报告）。"""

import argparse
import asyncio
import os
import sys

from data_builder import llm
from data_builder.config import SFT_OUTPUT_DIR
from data_builder.sources import get_sft_pairs, print_sft_summary
from data_builder.sft import run_all


def main():
    parser = argparse.ArgumentParser(description="Construct SFT data from teacher model")
    parser.add_argument("--model", default=None,
                        help="Model name (default: $VLLM_MODEL or auto-detect)")
    parser.add_argument("--base-url", default="http://localhost:8002",
                        help="vLLM server base URL")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of parallel requests")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries per record")
    args = parser.parse_args()

    model_name = args.model or os.getenv("VLLM_MODEL") or llm.fetch_available_model(args.base_url)
    if not model_name:
        print("FAIL: unable to determine model. Pass --model or set $VLLM_MODEL.")
        sys.exit(1)
    print(f"Using model: {model_name}")

    llm.init(args.base_url, model_name, args.retries)

    ebd_pairs, levir_pairs, second_pairs = get_sft_pairs()
    print("数据集统计：")
    print_sft_summary(ebd_pairs, levir_pairs, second_pairs)

    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_all(ebd_pairs, levir_pairs, second_pairs, args.concurrency))


if __name__ == "__main__":
    main()
