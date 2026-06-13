"""数据源抽象层：EBD/LEVIR/SECOND 数据集的 pair 构建与分配策略。

CoT/SFT 和 VQA 的分配规则集中在此模块，避免多脚本复刻。
"""

from collections import defaultdict
from pathlib import Path

from .config import (
    DATASET_ROOT,
    EBD_PER_EVENT,
    EBD_VQA_PER_EVENT,
    LEVIR_TAKE,
    SECOND_TAKE,
    VQA_MAX_PER_DATASET,
)


# ── Pair 构建 ──────────────────────────────────────────────


def build_ebd_pairs() -> list[dict]:
    """构建 EBD 数据集 pre/post 对。"""
    root = Path(DATASET_ROOT) / "EBD"
    pairs = []
    for event_dir in root.iterdir():
        if not event_dir.is_dir():
            continue
        event = event_dir.name
        img_dir = event_dir / "images"
        if not img_dir.exists():
            continue
        id_to_paths: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in img_dir.glob("*"):
            stem = img_path.stem
            parts = stem.rsplit("_", 3)
            if len(parts) < 4:
                continue
            img_id = parts[-3]
            if "pre" in stem:
                id_to_paths[img_id]["pre"] = img_path
            elif "post" in stem:
                id_to_paths[img_id]["post"] = img_path
        for img_id, paths in id_to_paths.items():
            if "pre" in paths and "post" in paths:
                pairs.append({
                    "event": event,
                    "pre": paths["pre"],
                    "post": paths["post"],
                })
    return pairs


def build_levir_pairs() -> list[dict]:
    """构建 LEVIR-CD+ 数据集 A/B 对。"""
    root = Path(DATASET_ROOT) / "LEVIR-CD+"
    pairs = []
    for split in ["train", "test"]:
        dir_a = root / split / "A"
        dir_b = root / split / "B"
        if not dir_a.exists() or not dir_b.exists():
            continue
        for img_path in sorted(dir_a.iterdir()):
            if not img_path.is_file():
                continue
            counterpart = dir_b / img_path.name
            if counterpart.exists():
                pairs.append({"pre": img_path, "post": counterpart})
    return pairs


def build_second_pairs() -> list[dict]:
    """构建 SECOND 数据集 im1/im2 对。"""
    root = Path(DATASET_ROOT) / "SECOND"
    pairs = []
    for split in ["train", "test"]:
        dir_im1 = root / split / "im1"
        dir_im2 = root / split / "im2"
        if not dir_im1.exists() or not dir_im2.exists():
            continue
        for img_path in sorted(dir_im1.iterdir()):
            if not img_path.is_file():
                continue
            counterpart = dir_im2 / img_path.name
            if counterpart.exists():
                pairs.append({"pre": img_path, "post": counterpart})
    return pairs


# ── 分配策略 ───────────────────────────────────────────────


def get_cot_image_set() -> set[str]:
    """返回 CoT/SFT 占用的 post 侧图像路径（绝对路径字符串）。"""
    cot_images: set[str] = set()

    ebd_all = build_ebd_pairs()
    ebd_by_event: dict[str, list[dict]] = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    for _event, pairs in sorted(ebd_by_event.items()):
        for p in pairs[:EBD_PER_EVENT]:
            cot_images.add(str(p["post"].resolve()))

    for p in build_levir_pairs()[:LEVIR_TAKE]:
        cot_images.add(str(p["post"].resolve()))

    for p in build_second_pairs()[:SECOND_TAKE]:
        cot_images.add(str(p["post"].resolve()))

    return cot_images


def get_sft_pairs() -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (ebd_pairs, levir_pairs, second_pairs)，已按采样参数截取。"""
    ebd_all = build_ebd_pairs()
    ebd_by_event: dict[str, list[dict]] = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    ebd_pairs = []
    for _event, pairs in sorted(ebd_by_event.items()):
        ebd_pairs.extend(pairs[:EBD_PER_EVENT])

    levir_pairs = build_levir_pairs()[:LEVIR_TAKE]
    second_pairs = build_second_pairs()[:SECOND_TAKE]

    return ebd_pairs, levir_pairs, second_pairs


def collect_vqa_candidates() -> dict[str, list[str]]:
    """收集 VQA 可用的 post 侧图像（排除 CoT 占用），按数据集分组。

    EBD 按事件子目录分别采样（每个事件取 EBD_VQA_PER_EVENT 张），
    LEVIR/SECOND 全量扫描后统一截取 VQA_MAX_PER_DATASET 张。

    返回 {dataset_name: [abs_image_paths]}。
    """
    cot_images = get_cot_image_set()
    result: dict[str, list[str]] = {}

    # ── EBD：按事件目录分别采样 ──
    ebd_root = Path(DATASET_ROOT) / "EBD"
    ebd_sampled: list[str] = []
    for event_dir in sorted(ebd_root.iterdir()):
        if not event_dir.is_dir():
            continue
        img_dir = event_dir / "images"
        if not img_dir.exists():
            continue
        event_imgs: list[str] = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                if "post" in img_path.stem:
                    event_imgs.append(str(img_path.resolve()))
        available = [p for p in event_imgs if p not in cot_images]
        ebd_sampled.extend(available[:EBD_VQA_PER_EVENT])
    result["EBD"] = ebd_sampled

    # ── LEVIR-CD+ ──
    levir_imgs: list[str] = []
    levir_root = Path(DATASET_ROOT) / "LEVIR-CD+"
    for split in ["train", "test"]:
        d = levir_root / split / "B"
        if not d.exists():
            continue
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                levir_imgs.append(str(img_path.resolve()))
    levir_available = [p for p in levir_imgs if p not in cot_images]
    result["LEVIR-CD+"] = levir_available[:VQA_MAX_PER_DATASET]

    # ── SECOND ──
    second_imgs: list[str] = []
    second_root = Path(DATASET_ROOT) / "SECOND"
    for split in ["train", "test"]:
        d = second_root / split / "im2"
        if not d.exists():
            continue
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                second_imgs.append(str(img_path.resolve()))
    second_available = [p for p in second_imgs if p not in cot_images]
    result["SECOND"] = second_available[:VQA_MAX_PER_DATASET]

    return result


def print_sft_summary(ebd_pairs, levir_pairs, second_pairs):
    """打印 SFT 数据集统计信息。"""
    ebd_all = build_ebd_pairs()
    ebd_by_event = defaultdict(list)
    for p in ebd_all:
        ebd_by_event[p["event"]].append(p)
    for event, pairs in sorted(ebd_by_event.items()):
        print(f"  EBD | {event}: {len(pairs)} 对可用, "
              f"取前 {min(len(pairs), EBD_PER_EVENT)} 对")
    print(f"  LEVIR-CD+: {len(build_levir_pairs())} 对可用, 取前 {len(levir_pairs)} 对")
    print(f"  SECOND:    {len(build_second_pairs())} 对可用, 取前 {len(second_pairs)} 对")
    print(f"  合计待处理: {len(ebd_pairs) + len(levir_pairs) + len(second_pairs)} 对")


def print_vqa_summary(per_dataset: dict[str, list[str]]):
    """打印 VQA 数据集统计信息。"""
    cot_images = get_cot_image_set()

    # EBD: 按事件统计
    ebd_root = Path(DATASET_ROOT) / "EBD"
    ebd_total = 0
    for event_dir in sorted(ebd_root.iterdir()):
        if not event_dir.is_dir():
            continue
        img_dir = event_dir / "images"
        if not img_dir.exists():
            continue
        event_post = [str(p.resolve()) for p in img_dir.glob("*")
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                      and "post" in p.stem]
        cot_in_event = sum(1 for p in event_post if p in cot_images)
        avail = len(event_post) - cot_in_event
        ebd_total += len(event_post)
        print(f"  EBD | {event_dir.name}: post图 {len(event_post)} 张, "
              f"CoT占用 {cot_in_event} 张, VQA可用 {avail} 张, "
              f"采样 {min(avail, EBD_VQA_PER_EVENT)} 张")
    print(f"  EBD 合计采样: {len(per_dataset['EBD'])} 张")

    # LEVIR / SECOND
    for root_name, label in [("LEVIR-CD+", "LEVIR-CD+"), ("SECOND", "SECOND")]:
        all_imgs = _count_post_images(root_name, label)
        cot_count = sum(1 for p in all_imgs if p in cot_images)
        available = len(all_imgs) - cot_count
        sampled = len(per_dataset[label])
        print(f"  {label}: post图 {len(all_imgs)} 张, CoT占用 {cot_count} 张, "
              f"VQA可用 {available} 张, 采样 {sampled} 张")


def _count_post_images(dataset_root_name: str, dataset_label: str) -> list[str]:
    """内部统计用：返回某数据集所有 post 图路径（不去重 CoT）。"""
    imgs: list[str] = []
    root = Path(DATASET_ROOT) / dataset_root_name
    if dataset_label == "EBD":
        for event_dir in root.iterdir():
            if not event_dir.is_dir():
                continue
            img_dir = event_dir / "images"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    if "post" in img_path.stem:
                        imgs.append(str(img_path.resolve()))
    elif dataset_label == "LEVIR-CD+":
        for split in ["train", "test"]:
            d = root / split / "B"
            if not d.exists():
                continue
            for img_path in d.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    imgs.append(str(img_path.resolve()))
    elif dataset_label == "SECOND":
        for split in ["train", "test"]:
            d = root / split / "im2"
            if not d.exists():
                continue
            for img_path in d.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    imgs.append(str(img_path.resolve()))
    return imgs
