from collections import defaultdict
from pathlib import Path
from datetime import datetime
import re
import json

from openai import OpenAI
from tqdm import tqdm


def build_disaster_registry(root_path):
    # 数据示例：
    # 这是文件夹的格式
    # /home/charles/mycode/sft+rl/dataset/EBD/EARTHQUAKE-TURKEY/images/
    # EARTHQUAKE - TURKEY_000013_post_disaster.png
    root = Path(root_path)
    # 构建三级嵌套字典并统计
    disaster_data = defaultdict(lambda: defaultdict(dict))
    # 关于匹配规则的解释如下：
    """
    ^ 在正则表达式中被称为 “行首锚点”。
    作用：它强制要求匹配必须从字符串的第一个字符开始。
    """

    """
    .* ：匹配任意文本
    .(点号)：代表 “任意单个字符”（除了换行符）。它可以是字母、数字、下划线、甚至是空格。
    * 默认是贪婪的 (Greedy)。贪婪的意思是：它会尽可能多地匹配字符，直到剩下的字符刚好能满足后续的模式。
    """

    # \d：代表 Digit（数字）。它等同于 [0-9]，即匹配 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 中的任意一个。
    # +：这是一个数量限定符，表示 “一个或多个”。
    # \d+：匹配数字串

    pattern = re.compile(r'^(.*)_(\d+)_(pre|post)_disaster')

    print("正在扫描文件系统...")
    # 在 root_path 目录下（包括所有子文件夹）寻找所有存放在 images 文件夹里的 .jpg 或 .png 文件
    all_imgs = list(root.glob("**/images/*.jpg")) + list(root.glob("**/images/*.png"))

    for img_path in all_imgs:
        """
        img_path.name: 得到 Hurricane_001_pre_disaster.jpg,带后缀。
        img_path.stem: 得到 Hurricane_001_pre_disaster,不带 .jpg。
        img_path.suffix: 得到 .jpg。
        """
        match = pattern.match(img_path.stem)
        if match:
            event, img_id, status = match.groups()
            disaster_data[event][img_id][status] = img_path
        # 按照三层字典放入，示例数据如下
        """
        {
            "Hurricane_Ian": {          # 第一级：事件
                "001": {                # 第二级：图片编号
                    "pre":  PosixPath("/.../Hurricane_Ian_001_pre_disaster.jpg"),  # 第三级：存入路径
                    "post": PosixPath("/.../Hurricane_Ian_001_post_disaster.jpg")
                }
            }
        }
        """

    # 统计并扁平化任务列表
    # {'灾害类型 (Event)':<30}：为第一列预留 30 字符。{'场景总数 (IDs)':<15}：为第二列预留 15 字符。
    print(f"\n{'灾害类型 (Event)':<30} | {'场景总数 (IDs)':<15} | {'文件总数 (Files)'}")
    print("-" * 70)

    task_list = []

    for event, id_dict in disaster_data.items():
        # s 可能是：{"pre": path1, "post": path2}
        # len(s)：计算这个三级字典的大小。
        # 如果pre和post都在，长度就是2。如果只有一张图，长度就是1。
        # for s in id_dict.values()：遍历该事件下所有的img_id，挨个计算它们拥有的文件数。
        # sum(...)：把这些数字全部加起来，得到该事件下的总文件数。
        total_files = sum(len(s) for s in id_dict.values())
        print(f"{event:<30} | {len(id_dict):<15} | {total_files}")
        # 计数器初始化：每换一种灾害，重新从 0 开始数
        event_added = 0
        for img_id, paths in id_dict.items():
            if event_added >= 100:  # 够 100 对了，后面的不要了
                break
            if 'pre' in paths and 'post' in paths:
                task_list.append({
                    'event': event,  # 这里的 event 会被填入 Prompt
                    'id': event_added,
                    'pre': paths['pre'],
                    'post': paths['post']
                })
                event_added += 1

        print(f"{event:<30} | {len(id_dict):<15} | 总进度: {len(task_list)}")
    return task_list


def main():
    dataset_root = "/home/charles/mycode/sft+rl/dataset/EBD"
    folder_name = Path(dataset_root).name
    current_time = datetime.now().strftime("%m.%d.%H:%M")
    output_file = f"{current_time}_{folder_name}_sft.jsonl"
    tasks = build_disaster_registry(dataset_root)
    if not tasks: return
    try:
        import os
        api_base = os.getenv('GENRM_API_BASE', 'http://localhost:8001/v1')
        temperature = float(os.getenv('GENRM_TEMPERATURE', '0.3'))
        client = OpenAI(
            api_key='EMPTY',
            base_url=api_base,
        )
        model_name = client.models.list().data[0].id
        import random

        with open(output_file, "w", encoding="utf-8") as f_out:
            # pbar = tqdm(tasks)：初始化进度条
            # pbar是"ProgressBar"的缩写，作为一个迭代器对象，它不仅能控制循环，还能让你在循环内部实时更新它的显示内容。
            pbar = tqdm(tasks)
            id = 0
            for task in pbar:
                id += 1
                event = task['event']
                img_id = task['id']
                pbar.set_description(f"处理中: {event} | ID: {img_id}")

                # 在循环内部
                # 2. 定义多种提问风格
                styles = [
                    "口语化，多用语气助词（如：那个、呃、呀）",
                    "极简主义，像搜索关键词一样简洁",
                    "语气急促，表现出强烈的紧迫感",
                    "学术严谨，使用地理学或地质学专业术语",
                    "现场目击者语气，带有真实感和描述性",
                    "新手小白，完全不懂行，用最通俗的话说",
                    "好奇探究，不仅问结论还问判断依据",
                    "社交媒体风，带点感慨或互动感",
                    "决策求助型，急切想知道接下来的具体行动方案",
                    "冷淡、克制、公事公办的语气",
                    "受灾群众身份，语气中充满对家园的关切",
                    "专业救援队员，干练、重点明确、强调实效",
                    "媒体记者采访语气，措辞正式且全面",
                    "非常有礼貌，带有大量的请、谢谢、麻烦了",
                    "结构化思维，喜欢分点（1. 2. 3.）提出疑问",
                    "行政管理风格，严肃、正式、要求汇报结论",
                ]
                selected_style = random.choice(styles)
                user_instruction = "请通过这两张对比图判断灾害类型，并根据受灾情况给出针对性的灾后恢复或行政处理建议"
                prompt = f"""
                原始指令：'{user_instruction}'\n
                要求：请将上述指令改写为一个真实的用户提问。你的提问风格要求：{selected_style}。
                注意：必须保留原指令中“判断灾害类型”和“给出建议”的核心意图。只返回改写后的问题文本，不要解释。"
                """
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": prompt},
                    ]
                }
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[message],
                    temperature=0.9,
                    max_tokens=100,
                )
                response_dict = response.choices[0].message.content
                record = {
                    "id": id,
                    "images": [
                        str(task['pre'].resolve()),
                        str(task['post'].resolve())
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": response_dict
                        }
                    ]
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
    except Exception as e:
        print(f"\n[错误] {event}_{img_id}: {e}")


if __name__ == "__main__":
    main()
