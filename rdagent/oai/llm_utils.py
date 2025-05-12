from __future__ import annotations

from typing import Any, Type

import numpy as np

from rdagent.core.utils import import_class
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash  # for compatible with previous import

import json
import re

def clean_json_string(text):
    # 去掉多余的结尾逗号
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # 处理非法反斜线（如 LaTeX 中的 \\），先保留双反斜线避免误伤
    text = text.replace('\\\\', '\\\\')  # 保留合法的双斜线
    text = text.replace('\\', '\\\\')   # 其它 \ 替换为 \\
    return text


def extract_json_objects(text) -> list[str]:
    results = []
    stack = []
    start_index = None
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                start_index = i
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
                if not stack and start_index is not None:
                    raw_json = text[start_index:i+1]

                    # 尝试原始解析
                    try:
                        parsed = json.loads(raw_json)
                        results.append(json.dumps(parsed))
                        continue
                    except json.JSONDecodeError:
                        pass

                    # 尝试 repr 的解析方式
                    try:
                        # 使用 repr 再去掉首尾引号（避免双重包裹）
                        escaped = repr(raw_json)[1:-1]
                        parsed = json.loads(escaped)
                        results.append(json.dumps(parsed))
                    except json.JSONDecodeError:
                        pass

                    start_index = None

    return results

def extract_json_objects11(text: str) -> list[str]:
    results: list[str] = []
    stack = []
    start_index = None
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                start_index = i  # potential JSON start
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
                if not stack and start_index is not None:
                    json_str = text[start_index:i+1]
                    try:
                        results.append(json.dumps(json.loads(json_str)))
                    except json.JSONDecodeError:
                        pass
                    start_index = None
    return results


def calculate_embedding_distance_between_str_list(
    source_str_list: list[str],
    target_str_list: list[str],
) -> list[list[float]]:
    if not source_str_list or not target_str_list:
        return [[]]

    embeddings = APIBackend().create_embedding(source_str_list + target_str_list)

    source_embeddings = embeddings[: len(source_str_list)]
    target_embeddings = embeddings[len(source_str_list) :]

    source_embeddings_np = np.array(source_embeddings)
    target_embeddings_np = np.array(target_embeddings)

    source_embeddings_np = source_embeddings_np / np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
    target_embeddings_np = target_embeddings_np / np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
    similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)

    return similarity_matrix.tolist()  # type: ignore[no-any-return]


def get_api_backend(*args: Any, **kwargs: Any) -> BaseAPIBackend:  # TODO: import it from base.py
    """
    get llm api backend based on settings dynamically.
    """
    api_backend_cls: Type[BaseAPIBackend] = import_class(LLM_SETTINGS.backend)
    print(f'------------ get_api_backend -- args:{args}, kwargs:{kwargs} -- LLM_SETTINGS: {LLM_SETTINGS}')
    return api_backend_cls(*args, **kwargs)


# Alias
APIBackend = get_api_backend
