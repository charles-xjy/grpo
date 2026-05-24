"""断点续传工具：从 JSONL 文件加载已完成记录。"""

import json
from pathlib import Path


def load_completed(filepath: Path) -> set[str]:
    """读取已完成的 images[0] 集合，用于断点续传。"""
    done = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["images"][0])
                except Exception:
                    pass
    return done


def load_desc_map(filepath: Path) -> dict[str, str]:
    """从 desc JSONL 加载 {images[0]: assistant_content} 映射。"""
    desc_map = {}
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    desc_map[rec["images"][0]] = rec["messages"][1]["content"]
                except Exception:
                    pass
    return desc_map
