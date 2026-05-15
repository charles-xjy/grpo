#!/usr/bin/env python3
"""Test vLLM multimodal — EBD dataset record 1."""

import base64
import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


EBD_TEACHER_PROMPT = """# Role
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
2. **客观性**：严禁过度推测。若影像模糊，必须使用"疑似"、"迹象显示"或"推测可能"等词汇。
3. **简洁性**：直接输出分析内容，禁止输出任何开场白、解释性文字或结束语（如"好的"、"我为您分析"等）。
4. **针对性**：所有分析必须锁定图像中的具体地物（如：指明"影像中部的矩形耕地"、"右侧的红色屋顶建筑"），拒绝通用化模板。"""



def main():
    base_url = "http://localhost:8001"
    model = "Qwen/Qwen3.5-27B"

    img1 = "/home/charles/mycode/sft+rl/dataset/EBD/EARTHQUAKE-TURKEY/images/EARTHQUAKE-TURKEY_002441_pre_disaster.png"
    img2 = "/home/charles/mycode/sft+rl/dataset/EBD/EARTHQUAKE-TURKEY/images/EARTHQUAKE-TURKEY_002441_post_disaster.png"

    system_prompt = EBD_TEACHER_PROMPT

    b64_1 = _encode_image(img1)
    b64_2 = _encode_image(img2)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_2}"}},
                    {"type": "text", "text": "请根据 system prompt 的要求，对比分析这两张灾前灾后影像。"},
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    print(f"Testing multimodal at {base_url} ...\n")

    try:
        req = Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
            elapsed = time.time() - t0

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        finish = choice.get("finish_reason", "?")

        print(f"elapsed={elapsed:.1f}s  tokens={usage.get('completion_tokens','?')}/{usage.get('total_tokens','?')}  finish={finish}\n")

        if reasoning:
            print("=" * 60)
            print("[REASONING]")
            print(reasoning)

        if content:
            print("=" * 60)
            print("[RESPONSE]")
            print(content)

        print("=" * 60)
        sys.exit(0)

    except URLError as e:
        print(f"FAIL: {e}")
        if hasattr(e, "read"):
            print(e.read().decode()[:2000])
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
