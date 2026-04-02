"""大模型调用服务（DeepSeek）。"""

import json
from dataclasses import dataclass
from time import sleep

import httpx


class LLMServiceError(RuntimeError):
    """大模型调用失败或响应解析失败时抛出。"""


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """大模型调用配置。

    Attributes:
        api_key: DeepSeek API Key。
        base_url: DeepSeek 基础地址。
        model: 使用的模型名。
        timeout_seconds: 单次请求超时时间（秒）。
        max_retries: 失败重试次数。
    """

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端。"""

    def __init__(self, config: LLMConfig) -> None:
        """初始化客户端。

        Args:
            config: 大模型调用配置。
        """
        self.config = config

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """请求模型生成回复。

        Args:
            messages: OpenAI 兼容格式消息列表。
            temperature: 采样温度。

        Returns:
            str: 模型返回的文本内容。
        """
        # API Key 为空直接报错，避免无意义网络请求。
        if not self.config.api_key:
            raise LLMServiceError("DEEPSEEK_API_KEY 未配置")

        # 构造目标 URL。
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        # 构造请求头。
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        # 构造请求体。
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }

        last_error: Exception | None = None
        # 最多执行 max_retries + 1 次（含首次）。
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return _extract_content(data)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                # 到达最后一次尝试则退出循环。
                if attempt >= self.config.max_retries:
                    break
                # 线性退避，避免瞬时反复请求。
                sleep(0.5 * (attempt + 1))

        raise LLMServiceError(f"DeepSeek 调用失败: {last_error}") from last_error


def _extract_content(data: object) -> str:
    """从 DeepSeek 响应 JSON 中提取回答文本。

    Args:
        data: HTTP 响应反序列化后的对象。

    Returns:
        str: `choices[0].message.content` 去首尾空白后的值。
    """
    try:
        if not isinstance(data, dict):
            raise ValueError("响应不是 JSON 对象")
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices 为空")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("choices[0] 格式错误")
        message = first_choice["message"]
        if not isinstance(message, dict):
            raise ValueError("message 格式错误")
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("content 不是字符串")
        return content.strip()
    except Exception as exc:  # noqa: BLE001
        serialized = json.dumps(data, ensure_ascii=False)
        raise LLMServiceError(f"无法解析大模型响应: {serialized}") from exc
