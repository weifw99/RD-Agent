from __future__ import annotations

from typing import Any, Type

import numpy as np

from rdagent.core.utils import import_class
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash  # for compatible with previous import

import json
import re

import re


def remove_tag_with_content(text: str, tag: str = "think") -> str:
    """
    移除指定标签及其内部的所有内容（支持嵌套）。

    :param text: 原始字符串
    :param tag: 要移除的标签名，如 'think' 或 'file'
    :return: 清洗后的字符串
    """
    # 匹配标签开始和结束的正则表达式
    pattern = rf"<{tag}.*?>.*?</{tag}>"  # 匹配 <think>...</think>
    pattern += "|" + rf"<{tag}.*?/>"  # 匹配自闭合标签 <think ... />
    pattern += "|" + rf"<{tag}[^>]*>"  # 匹配没有闭合标签的情况

    # 使用正则替换为空字符串
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text.strip()


def parse_json_result(result_str) -> str:
    """解析执行结果列中的JSON字符串，支持处理带有标签的复杂文本格式"""
    if not result_str or not isinstance(result_str, str):
        return ''

    try:
        # 移除<think>标签
        result_str = remove_tag_with_content(result_str, 'think')

        # 尝试查找并提取JSON文本
        # 首先尝试提取```json标记中的内容
        if '```json' in result_str:
            json_start = result_str.find('```json') + 7
            # 跳过可能存在的换行符
            while json_start < len(result_str) and result_str[json_start] in ['\n', '\r', ' ']:
                json_start += 1
            json_end = result_str.find('```', json_start)
            if json_end == -1:
                json_end = len(result_str)
            json_text = result_str[json_start:json_end].strip()
        # 然后尝试提取{开头}结尾的JSON文本
        elif result_str.strip().startswith('{') and result_str.strip().endswith('}'):
            json_text = result_str.strip()
        # 最后尝试在文本中查找完整的JSON对象
        else:
            start_idx = result_str.find('{')
            if start_idx != -1:
                # 找到匹配的结束括号
                count = 1
                end_idx = start_idx + 1
                while count > 0 and end_idx < len(result_str):
                    if result_str[end_idx] == '{':
                        count += 1
                    elif result_str[end_idx] == '}':
                        count -= 1
                    end_idx += 1
                if count == 0:
                    json_text = result_str[start_idx:end_idx].strip()
                else:
                    return ''
            else:
                return ''

        # 清理JSON文本中的特殊字符
        json_text = json_text.strip().replace('\xa0', ' ')

        return json_text
    except (json.JSONDecodeError, TypeError, AttributeError, IndexError) as e:
        # 如果解析失败，返回空字典
        return ''



# def clean_json_str(text: str) -> str:
#     # 移除尾随逗号（包括对象和数组中的）
#     text = re.sub(r',\s*([}\]])', r'\1', text)
#     return text

def clean_json_str(json_str: str) -> str:
    """
    尽可能清洗 JSON 字符串中的非标准格式问题，使其可被 json.loads 正确解析
    """
    cleaned = json_str.strip()

    # 修复非标准布尔值
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)

    # 删除 key-value 后多余逗号（在 } 或 ] 之前）
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

    # 修复单引号包裹的 JSON 为双引号
    def replace_single_quotes(match):
        inner = match.group(0)
        return inner.replace("'", '"')
    cleaned = re.sub(r"'{.*?}'", replace_single_quotes, cleaned)

    # 修复 value 中的裸引号（比如 `"key": "value with "quote""`）
    def escape_inner_quotes(m):
        key, val = m.group(1), m.group(2)
        # 用占位符规避公式影响再反解
        val = val.replace('\\"', '"')  # 先还原错误转义
        val = re.sub(r'(?<!\\)"', r'\\"', val)  # 再重新转义裸引号
        return f'"{key}": "{val}"'
    cleaned = re.sub(r'"([^"]+)":\s*"((?:.|\n)*?)"', escape_inner_quotes, cleaned)

    return cleaned


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

                    # 判断是否是这种结构： { "code": "import pandas as pd"}
                    import re
                    # 提取 code 内容（中间双引号内的部分）
                    match = re.search(r'"code": "(.*)"\s*}', raw_json, re.DOTALL)
                    if match:
                        code_raw = match.group(1)

                        # 替换字符串中的特殊字符为合法 JSON 字符串格式
                        code_fixed = code_raw.replace('\\', '\\\\') \
                            .replace('"', '\\"') \
                            .replace('\n', '\\n')

                        # 构造合法 JSON 字符串
                        json_fixed_str = '{"code": "' + code_fixed + '"}'
                        results.append(json_fixed_str)

                        # 解析为 dict
                        # data = json.loads(json_fixed_str)
                        # json_str = json.dumps(data)
                        # results.append(json_str)
                    else:
                        pass

                    start_index = None

    return results

import json
import re

def safe_parse_value(val):
    """
    安全地解析值：如果是嵌套的 JSON 字符串，尝试递归解析；否则返回原始或清洗后的值。
    """
    if isinstance(val, str):
        # 尝试去除首尾引号包裹并进行解转义
        val = val.strip()
        try:
            # 修复可能错误的单引号为双引号
            fixed = val.replace("'", '"')
            # 替换 JSON 中的特殊布尔值（如 True/False/null）
            fixed = re.sub(r'\b(True|False)\b', lambda m: m.group(0).lower(), fixed)
            fixed = re.sub(r'\bNone\b', 'null', fixed)
            return parse_nested_json(fixed)
        except Exception:
            pass
    return val

def parse_nested_json(data):
    """
    递归解析嵌套 JSON，返回结构化数据。
    """
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return parse_nested_json(parsed)
        except Exception:
            return data
    elif isinstance(data, dict):
        return {str(k): parse_nested_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [parse_nested_json(item) for item in data]
    else:
        return data

def transform_json_string(input_str):
    """
    主方法：输入嵌套 JSON 字符串，返回标准 JSON 字符串。
    """
    try:
        cleaned_data = parse_nested_json(input_str)
        return json.dumps(cleaned_data, ensure_ascii=False, indent=2)
    except Exception as e:
        print("解析失败：", e)
        return None


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
