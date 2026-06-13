"""LLM 调用封装：AsyncOpenAI + 重试 + 指数退避。支持多客户端实例（如教师/学生模型）。"""

import asyncio
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

from openai import AsyncOpenAI

_default_client: "LLMClient" = None


class LLMClient:
    def __init__(self, base_url: str, model_name: str, max_retries: int = 3):
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")
        self.model_name = model_name
        self.max_retries = max_retries

    async def _call_with_retry(self, coro_factory, desc: str = ""):
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} retries ({desc}): {e}")

    async def chat_with_images(
        self,
        prompt: str,
        image_paths: list[str | Path],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            b64 = encode_image(img_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        desc = str(Path(image_paths[0]).stem) if image_paths else "image_chat"
        return await self._call_with_retry(_call, desc)

    async def chat_text_only(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        return await self._call_with_retry(_call, prompt[:80])


def init(base_url: str, model_name: str, max_retries: int = 3) -> None:
    global _default_client
    _default_client = LLMClient(base_url, model_name, max_retries)


def get_model_name() -> str:
    return _default_client.model_name if _default_client else ""


def encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fetch_available_model(base_url: str) -> str:
    req = Request(f"{base_url}/v1/models", headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return ""


async def chat_with_images(*args, **kwargs):
    return await _default_client.chat_with_images(*args, **kwargs)


async def chat_text_only(*args, **kwargs):
    return await _default_client.chat_text_only(*args, **kwargs)
