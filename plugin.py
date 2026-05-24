import asyncio
import os
import random
import re
import textwrap
import base64
from typing import List
from swift.rewards import AsyncORM, orms
from swift.utils import get_logger

logger = get_logger()

def encode_image(image_path):
    """将物理路径图片转为 base64。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class RemoteSensingGMDGReward(AsyncORM):
    """
    遥感解译 GRPO 奖励函数：
    1. 格式奖励 (30%, 3.0分): 检查结构标题、负面约束。
    2. 内容奖励 (70%, 7.0分): 结合图像检查幻觉、准确性、空间锚定。
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        from openai import OpenAI
        self.api_base = os.getenv('GENRM_API_BASE', 'http://localhost:8002/v1')
        self.temperature = float(os.getenv('GENRM_TEMPERATURE', '0.3'))
        self.sem = asyncio.Semaphore(int(os.getenv('GENRM_CONCURRENCY', '8')))

        try:
            self.client = OpenAI(api_key='EMPTY', base_url=self.api_base)
            self.model_name = self.client.models.list().data[0].id
            logger.info(f'GMDGReward initialized with model: {self.model_name}')
        except Exception as e:
            raise RuntimeError(f'Failed to connect to judge model at {self.api_base}: {e}')

        self.dimensions = [
            {
                'name': '格式与结构奖励',
                'max_score': 3.0,
                'use_image': False,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位文档格式与表达风格审查专家。请检查学生的“遥感解译报告”是否符合以下规定：

                    ## 评分准则 (满分 3.0)
                    1. **标题完备性 (2.0分)**：报告必须包含结构化标题（如“## 1. 总体概述”、“## 2. 详细分析”等）。标题必须清晰且逻辑层次分明。
                    2. **人设口吻匹配 (1.0分)**：检查回答的口吻是否符合问题中要求的身份（如农民、专家、学生等）。**允许并鼓励**符合身份的自然开场白。

                    ## 输出格式
                    仅输出分数：[[x.x]]
                """),
            },
            {
                'name': '内容准确性与图像锚定',
                'max_score': 7.0,
                'use_image': True,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位资深的遥感解译专家。请结合提供的“图像”对“学生报告”进行真伪校验。

                    ## 评分准则 (满分 7.0)
                    1. **真实性 (4.0分)**：报告中描述的变化是否真实存在于图中？是否存在明显的幻觉（如图中没水却说水体扩张）？
                    2. **空间锚定 (3.0分)**：分析是否锁定了具体地物位置（如“影像左上角”、“紧邻道路的建筑”）？描述越具体分越高。

                    ## 输出格式
                    给出理由，最后输出分数：[[x.x]]
                """),
            },
        ]

    def _extract_score(self, response: str, max_score: float) -> float:
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
        if match:
            return min(max(float(match.group(1)), 0.0), max_score)
        return 0.0

    async def _score_dimension(self, session, question: str, completion: str, images: list, dim: dict) -> float:
        import aiohttp
        
        # 构造 User Content
        user_content = [{"type": "text", "text": f"问题：{question}\n\n待评报告：\n{completion}"}]
        
        # 如果该维度需要看图，则加入图像 base64
        if dim['use_image'] and images:
            for img_path in images:
                if os.path.exists(img_path):
                    b64 = encode_image(img_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })

        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': dim['system_prompt']},
                {'role': 'user', 'content': user_content}
            ],
            'temperature': self.temperature,
            'max_tokens': 512,
        }

        try:
            async with self.sem:
                async with session.post(f'{self.api_base}/chat/completions', json=payload, timeout=120) as resp:
                    if resp.status != 200: return 0.0
                    result = await resp.json()
                    res_text = result['choices'][0]['message']['content']
                    return self._extract_score(res_text, dim['max_score'])
        except:
            return 0.0

    async def _score_single(self, session, question: str, completion: str, images: list) -> float:
        tasks = [self._score_dimension(session, question, completion, images, dim) for dim in self.dimensions]
        scores = await asyncio.gather(*tasks)
        total = sum(scores)
        logger.info(f'Sample Score - Format: {scores[0]:.1f}, Content: {scores[1]:.1f}, Total: {total:.1f}')
        return total

    async def __call__(self, completions, messages, **kwargs) -> List[float]:
        import aiohttp
        
        # 从 messages 中提取问题，从 kwargs 中提取图片路径
        questions = []
        for msg_list in messages:
            q = next((m['content'] for m in reversed(msg_list) if m['role'] == 'user'), '')
            questions.append(q)
            
        # Swift GRPO 传递的 images 通常在 kwargs['images'] 中
        all_images = kwargs.get('images', [[]] * len(completions))

        async with aiohttp.ClientSession() as session:
            tasks = [self._score_single(session, q, c, imgs) for q, c, imgs in zip(questions, completions, all_images)]
            rewards = await asyncio.gather(*tasks)
            return list(rewards)

orms['LLM_as_judger'] = RemoteSensingGMDGReward
