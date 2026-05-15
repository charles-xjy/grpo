import base64
import json
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from collections import defaultdict

# ================= 1. 配置部分 =================
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
model_name = "Qwen/Qwen3-VL-32B-Instruct"

dataset_root = "/home/charles/mycode/sft+rl/dataset/EBD"
folder_name = Path(dataset_root).name
current_time = datetime.now().strftime("%m.%d.%H:%M")
output_file = f"{current_time}_{folder_name}_sft.jsonl"

# 【修改点】在 Prompt 中预留一个 {event} 占位符，告诉模型已知的灾害背景
prompt_template = """
    # Context
    本次分析的灾害事件类型为：{event}。

    # Role
    你是一位资深的遥感地质专家与防灾减灾顾问。请对比分析提供的两张影像（图1灾前，图2灾后）。

    # Task
    请对比分析上传的两张图像，并严格按照以下结构输出技术报告：

    ## 1. 灾害类型判定与物理机制
    - **判定结论**：给出具体的灾害名称（如：走滑断层地表破裂、推覆构造、浅表层滑坡等）。
    - **特征描述**：结合影像中的光谱特征、纹理断裂及几何位移，描述灾害表现（如：地表破裂线走向、地表粗糙度变化、地物水平位移量估算）。
    - **成因简析**：简述地质或气象驱动机制。

    ## 2. 针对性损害评估
    - **基础设施**：指明具体受损点（如：道路某段的物理偏移、桥梁坍塌、电力塔基变形）。
    - **农业与生态**：分析具体斑块的受损情况（如：林地破碎化、农田灌溉渠断裂、生态斑块连通性受阻）。
    - **建筑安全**：针对影像中具体建筑的基底偏移、倒塌或疑似结构损伤进行说明。

    ## 3. 差异化防治建议
    - **工程避让**：根据破裂线或灾害影响范围，划定精确的建筑禁建区或缓冲区。
    - **韧性修复**：针对影像中损坏的具体设施提供修复方案（如：跨断层的柔性管道连接、加筋土路基复原）。
    - **监测布控**：建议在影像中受损严重的特定坐标点布设形变监测传感器。

    # Output Style (严格执行)
    1. **专业性**：必须使用灾害分析术语（如：地表形变矢量、生境破碎化、光谱差异、构造应力）。
    2. **客观性**：严禁过度推测。若影像模糊，必须使用“疑似”、“迹象显示”或“推测可能”等词汇。
    3. **简洁性**：直接输出分析内容，禁止输出任何开场白、解释性文字或结束语（如“好的”、“我为您分析”等）。
    4. **针对性**：所有分析必须锁定图像中的具体地物（如：指明“影像中部的矩形耕地”、“右侧的红色屋顶建筑”），拒绝通用化模板。
"""


# ================= 2. 工具函数 =================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_disaster_registry(root_path):
    """构建三级嵌套字典并统计"""
    root = Path(root_path)
    disaster_data = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r'^(.*)_(\d+)_(pre|post)_disaster')

    print("正在扫描文件系统...")
    all_imgs = list(root.glob("**/images/*.jpg")) + list(root.glob("**/images/*.png"))

    for img_path in all_imgs:
        match = pattern.match(img_path.stem)
        if match:
            event, img_id, status = match.groups()
            disaster_data[event][img_id][status] = img_path

    # 统计并扁平化任务列表
    print(f"\n{'灾害类型 (Event)':<30} | {'场景总数 (IDs)':<15} | {'文件总数 (Files)'}")
    print("-" * 70)

    task_list = []

    for event, id_dict in disaster_data.items():
        total_files = sum(len(s) for s in id_dict.values())
        print(f"{event:<30} | {len(id_dict):<15} | {total_files}")
        # 计数器初始化：每换一种灾害，重新从 0 开始数
        event_added = 0
        for img_id, paths in id_dict.items():
            if event_added >= 50:  # 够 100 对了，后面的不要了
                break
            if 'pre' in paths and 'post' in paths:
                task_list.append({
                    'event': event,  # 这里的 event 会被填入 Prompt
                    'id': img_id,
                    'pre': paths['pre'],
                    'post': paths['post']
                })
                event_added += 1

        print(f"{event:<30} | {len(id_dict):<15} | 总进度: {len(task_list)}")
    return task_list


# ================= 3. 主程序逻辑 =================

def main():
    tasks = build_disaster_registry(dataset_root)
    if not tasks: return

    with open(output_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(tasks)
        for task in pbar:
            event = task['event']
            img_id = task['id']
            pbar.set_description(f"处理中: {event} | ID: {img_id}")

            try:
                base64_pre = encode_image(task['pre'])
                base64_post = encode_image(task['post'])

                # 【修改点】动态填充 Prompt，将当前任务的 event 传入
                formatted_prompt = prompt_template.format(event=event)

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": formatted_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_pre}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_post}"}},
                            ],
                        }
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )

                answer = response.choices[0].message.content
                user_instruction = "你现在是一名灾后评估专家。请通过这两张对比图分析受灾情况并给出建议。"
                # 构造记录
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>\n<image>\n{user_instruction}"
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ],
                    "images": [
                        str(task['pre'].resolve()),
                        str(task['post'].resolve())
                    ]
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\n[错误] {event}_{img_id}: {e}")
            except KeyboardInterrupt:
                break

    print(f"\n结果已存入: {output_file}")


if __name__ == "__main__":
    main()
