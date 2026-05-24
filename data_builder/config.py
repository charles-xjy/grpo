"""全局配置：路径、采样参数。"""

from pathlib import Path

DATASET_ROOT = "/home/charles/mycode/sft+rl/dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft_data"
VQA_OUTPUT_DIR = PROJECT_ROOT / "data" / "vqa_data"

# CoT/SFT 采样参数
EBD_PER_EVENT = 50
LEVIR_TAKE = 200
SECOND_TAKE = 500

# VQA 采样上限
VQA_MAX_PER_DATASET = 100       # LEVIR / SECOND 各取前 N 张
EBD_VQA_PER_EVENT = 20          # EBD 每个事件目录各取 N 张（共 7 个事件）
