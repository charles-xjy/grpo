import asyncio
import os
import random
import re
import textwrap
from typing import List
from swift.rewards import AsyncORM, orms
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py
        --reward_funcs my_reward_function
"""


class AsyncGenRMReward(AsyncORM):
    """
    An async reward function that scores completions on 4 separate dimensions,
    each making an independent API call to a generative reward model.
    The 4 scores are summed to produce the final reward.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        from openai import OpenAI
        self.api_base = os.getenv('GENRM_API_BASE', 'http://localhost:8001/v1')
        self.temperature = float(os.getenv('GENRM_TEMPERATURE', '0.3'))

        # Initialize OpenAI client to get the model name
        try:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=self.api_base,
            )
            self.model_name = self.client.models.list().data[0].id
            logger.info(f'AsyncGenRMReward initialized with model: {self.model_name}')
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy --model <model_name> --port 8000 --infer_backend vllm'.") from e

        # 4 separate dimensions, each scored independently then summed
        self.dimensions = [
            {
                'name': '结构完备性',
                'max_score': 2.0,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位精通遥感图像解译与地质灾害评估的特级专家，负责对一份“地质灾害技术报告”进行严苛的打分。

                    # Evaluation Criterion (仅评分以下单一维度)
                    ## 结构完备性 (满分 2.0)
                    - 必须严格从三个方向进行分析：1. 灾害类型判定与物理机制、2. 针对性损害评估、3. 差异化防治建议。
                    - 缺失任一核心模块，本项记 0 分。

                    ## 输出格式
                    仅输出一个分数（0~2.0）：[[x.x]]
                """),
            },
            {
                'name': '图像锚定精确度',
                'max_score': 3.0,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位精通遥感图像解译与地质灾害评估的特级专家，负责对一份“地质灾害技术报告”进行严苛的打分。

                    # Evaluation Criterion (仅评分以下单一维度)
                    ## 图像锚定精确度 (满分 3.0)
                    - 【核心指标】分析是否锁定图像具体位置进行分析？（如：指明“影像左上角的白色厂房”“横穿中部的河流南岸”）。
                    - 若分析内容泛泛而谈（通用模板），本项记 0 分。

                    ## 输出格式
                    仅输出一个分数（0~3.0）：[[x.x]]
                """),
            },
            {
                'name': '术语使用与地质逻辑',
                'max_score': 3.0,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位精通遥感图像解译与地质灾害评估的特级专家，负责对一份“地质灾害技术报告”进行严苛的打分。

                    # Evaluation Criterion (仅评分以下单一维度)
                    ## 术语使用与地质逻辑 (满分 3.0)
                    - 是否准确应用了诸如“地表形变矢量”“光谱差异”“构造应力”等专业词汇。
                    - 逻辑必须符合地质灾害的物理成因。

                    ## 输出格式
                    仅输出一个分数（0~3.0）：[[x.x]]
                """),
            },
            {
                'name': '负面约束执行力',
                'max_score': 2.0,
                'system_prompt': textwrap.dedent("""\
                    # Role
                    你是一位精通遥感图像解译与地质灾害评估的特级专家，负责对一份“地质灾害技术报告”进行严苛的打分。

                    # Evaluation Criterion (仅评分以下单一维度)
                    ## 负面约束执行力 (满分 2.0)
                    - 严禁任何形式的开场白（如“好的”“报告如下”）或结束语。
                    - 必须保持绝对客观，不确定处必须使用“疑似”“迹象显示”。

                    ## 输出格式
                    仅输出一个分数（0~2.0）：[[x.x]]
                """),
            },
        ]

    def _build_eval_prompt(self, question: str, completion: str) -> str:
        """Build the evaluation prompt for the reward model."""
        return textwrap.dedent(f"""
            ## User Question
            {question}

            ## AI Assistant's Response
            {completion}

            ## Your Evaluation
            Evaluate the above response and provide your score.
        """).strip()

    def _extract_score(self, response: str, max_score: float) -> float:
        """Extract a single [[score]] from the response, clipped to [0, max_score]."""
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
        if match:
            return min(max(float(match.group(1)), 0.0), max_score)

        # Fallback: try to find any number at the end
        match = re.search(r'(\d+(?:\.\d+)?)\s*$', response.strip())
        if match:
            return min(max(float(match.group(1)), 0.0), max_score)

        logger.warning(f'Could not extract score from response: {response[:100]}...')
        return 0.0

    async def _score_dimension(self, session, question: str, completion: str, dim: dict) -> float:
        """Score a single completion on one dimension."""
        import aiohttp

        eval_prompt = self._build_eval_prompt(question, completion)

        payload = {
            'model': self.model_name,
            'messages': [{
                'role': 'system',
                'content': dim['system_prompt']
            }, {
                'role': 'user',
                'content': eval_prompt
            }],
            'temperature': self.temperature,
            'max_tokens': 512,
            'seed': random.randint(0, 1000000),
        }

        try:
            async with session.post(
                    f'{self.api_base}/chat/completions', json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f'API error {resp.status}: {error_text[:200]}')
                    return 0.0

                result = await resp.json()
                response_content = result['choices'][0]['message']['content']
                return self._extract_score(response_content, dim['max_score'])

        except asyncio.TimeoutError:
            logger.warning(f'API request timed out for dimension {dim["name"]}')
            return 0.0
        except Exception as e:
            logger.warning(f'Error calling reward model API for dimension {dim["name"]}: {e}')
            return 0.0

    async def _score_single(self, session, question: str, completion: str) -> float:
        """
        Score a single completion on all 4 dimensions in parallel, return the sum.
        """
        tasks = [
            self._score_dimension(session, question, completion, dim)
            for dim in self.dimensions
        ]
        dim_scores = await asyncio.gather(*tasks)

        # Log per-dimension scores for debugging
        for name, score in zip([d['name'] for d in self.dimensions], dim_scores):
            logger.info(f'Dim {name}: {score:.2f}')

        total = sum(dim_scores)
        logger.info(f'Total score: {total:.2f}')
        return total

    async def __call__(self, completions, messages, **kwargs) -> List[float]:
        """
        Score completions using a generative reward model via async API calls.

        Args:
            completions: List of model-generated responses
            messages: List of conversation messages (used to extract the question)
            **kwargs: Additional arguments (unused)

        Returns:
            List of reward scores in [0, 10] range (sum of 4 dimensions)
        """
        import aiohttp

        # Extract questions from messages (assuming the last user message is the question)
        questions = []
        for msg_list in messages:
            question = ''
            for msg in reversed(msg_list):
                if msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
            questions.append(question)

        async with aiohttp.ClientSession() as session:
            tasks = [self._score_single(session, q, c) for q, c in zip(questions, completions)]
            rewards = await asyncio.gather(*tasks)
            return list(rewards)


orms['LLM_as_judger'] = AsyncGenRMReward
