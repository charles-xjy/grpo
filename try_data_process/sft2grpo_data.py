import json


def reassign_all_ids(input_file, output_file):
    """
    读取 JSONL 文件，删除旧 ID 并根据行号重新分配 ID。
    """
    new_data_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for index, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                user_instruction = "请通过这两张对比图判断灾害类型，并根据受灾情况给出针对性的灾后恢复或行政处理建议"
                # 核心逻辑：删除旧 ID（如果存在），并赋予新的 index
                # 如果你希望从 1 开始，改为 index + 1
                new_item = {
                    "images": item["images"],
                    "messages": [
                        {
                            "role": "user",
                            "content": user_instruction
                        }
                    ]
                }

                # 保持中文不被转义，写入新文件
                f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')


            except json.JSONDecodeError:
                print(f"警告：跳过无效的 JSON 行 (第 {index} 行)")

    print(f"处理完成！共重写 {new_data_count} 条数据。")
    print(f"新文件路径: {output_file}")


# --- 配置路径 ---
# 请将 'dataset.jsonl' 替换为你实际的文件名
input_path = "/home/charles/mycode/grpo/04.20.16:31_EBD_sft.jsonl"
output_path = "grpo_dataset2.jsonl"

reassign_all_ids(input_path, output_path)
