import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Union
from lightrag.utils import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    import requests
    llm_api_url = "http://223.109.239.14:10000/v1/chat/completions"
    # 调用自托管聊天接口，构造标准OpenAI messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # 尝试将history_messages并入（若为list[dict]格式）
    if isinstance(history_messages, list):
        for msg in history_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
    # 当前用户问题
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": "qwen2__5-72b",
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "top_p": 1,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(llm_api_url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # 兼容常见返回格式
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        choice = data["choices"][0]
        # OpenAI-style
        if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        # Some providers use 'text'
        if "text" in choice:
            return choice["text"]
    raise RuntimeError(f"Unexpected LLM response format: {data}")
    # stream = True if kwargs.get("stream") else False
    #
    # kwargs.pop("max_tokens", None)
    # # kwargs.pop("response_format", None) # allow json
    # host = kwargs.pop("host", None)
    # timeout = kwargs.pop("timeout", None)
    # kwargs.pop("hashing_kv", None)
    # api_key = kwargs.pop("api_key", None)
    # headers = {
    #     "Content-Type": "application/json",
    #     "User-Agent": f"LightRAG/{__api_version__}",
    # }
    # if api_key:
    #     headers["Authorization"] = f"Bearer {api_key}"
    #
    # ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    #
    # try:
    #     messages = []
    #     if system_prompt:
    #         messages.append({"role": "system", "content": system_prompt})
    #     messages.extend(history_messages)
    #     messages.append({"role": "user", "content": prompt})
    #
    #     response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    #     if stream:
    #         """cannot cache stream response and process reasoning"""
    #
    #         async def inner():
    #             try:
    #                 async for chunk in response:
    #                     yield chunk["message"]["content"]
    #             except Exception as e:
    #                 logger.error(f"Error in stream response: {str(e)}")
    #                 raise
    #             finally:
    #                 try:
    #                     await ollama_client._client.aclose()
    #                     logger.debug("Successfully closed Ollama client for streaming")
    #                 except Exception as close_error:
    #                     logger.warning(f"Failed to close Ollama client: {close_error}")
    #
    #         return inner()
    #     else:
    #         model_response = response["message"]["content"]
    #
    #         """
    #         If the model also wraps its thoughts in a specific tag,
    #         this information is not needed for the final
    #         response and can simply be trimmed.
    #         """
    #
    #         return model_response
    # except Exception as e:
    #     try:
    #         await ollama_client._client.aclose()
    #         logger.debug("Successfully closed Ollama client after exception")
    #     except Exception as close_error:
    #         logger.warning(
    #             f"Failed to close Ollama client after exception: {close_error}"
    #         )
    #     raise e
    # finally:
    #     if not stream:
    #         try:
    #             await ollama_client._client.aclose()
    #             logger.debug(
    #                 "Successfully closed Ollama client for non-streaming response"
    #             )
    #         except Exception as close_error:
    #             logger.warning(
    #                 f"Failed to close Ollama client in finally block: {close_error}"
    #             )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    # api_key = kwargs.pop("api_key", None)
    # headers = {
    #     "Content-Type": "application/json",
    #     "User-Agent": f"LightRAG/{__api_version__}",
    # }
    # if api_key:
    #     headers["Authorization"] = f"Bearer {api_key}"
    #
    # host = kwargs.pop("host", None)
    # timeout = kwargs.pop("timeout", None)
    #
    # ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    # try:
    #     options = kwargs.pop("options", {})
    #     data = await ollama_client.embed(
    #         model=embed_model, input=texts, options=options
    #     )
    #     return np.array(data["embeddings"])
    import requests
    base_url = "http://10.25.20.246:6109/v1"  # 修正为根路径
    api_url = base_url.rstrip('/') + '/embeddings'
    payload = {"input": texts}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vectors = []
    for item in data.get("data", []):
        if "embedding" in item:
            vectors.append(item["embedding"])
    if not vectors:
        raise RuntimeError(f"Embeddings response invalid: {data}")
    return np.array(vectors, dtype=np.float32)

    # except Exception as e:
    #     logger.error(f"Error in ollama_embed: {str(e)}")
    #     try:
    #         await ollama_client._client.aclose()
    #         logger.debug("Successfully closed Ollama client after exception in embed")
    #     except Exception as close_error:
    #         logger.warning(
    #             f"Failed to close Ollama client after exception in embed: {close_error}"
    #         )
    #     raise e
    # finally:
    #     try:
    #         await ollama_client._client.aclose()
    #         logger.debug("Successfully closed Ollama client after embed")
    #     except Exception as close_error:
    #         logger.warning(f"Failed to close Ollama client after embed: {close_error}")
