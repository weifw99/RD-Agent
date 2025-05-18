


import json
from typing import Dict, Union, List

from pydantic import TypeAdapter

def test_json_parse1():
    test_str = '''
    
    {
        "RVI_10_30": {
            "description": "Relative Volatility Index compares short-term volatility (10-day) to long-term volatility (30-day) to capture abnormal price fluctuation patterns.",
            "formulation": "\\text{RVI} = \\frac{\\sigma_{\\text{short}}}{\\sigma_{\\text{long}}}, \\quad \\sigma_{\\text{short}} = \\text{STD}(\$close, 10), \\quad \\sigma_{\\text{long}} = \\text{STD}(\$close, 30)",
            "variables": {
                "STD($close, 10)": "10-day rolling standard deviation of closing prices",
                "STD($close, 30)": "30-day rolling standard deviation of closing prices"
            }
        },
        "RVI_5_20": {
            "description": "Alternative Relative Volatility Index with shorter-term (5-day) and longer-term (20-day) volatility comparison for more responsive regime detection.",
            "formulation": "\\text{RVI} = \\frac{\\sigma_{\\text{short}}}{\\sigma_{\\text{long}}}, \\quad \\sigma_{\\text{short}} = \\text{STD}(\$close, 5), \\quad \\sigma_{\\text{long}} = \\text{STD}(\$close, 20)",
            "variables": {
                "STD($close, 5)": "5-day rolling standard deviation of closing prices",
                "STD($close, 20)": "20-day rolling standard deviation of closing prices"
            }
        }
    }
    
    '''


    print( json.loads(test_str) )

def test_json_parse2():

    import json

    test_str1 = r'''
    
    {
        "RVI_10_30": {
            "description": "Relative Volatility Index compares short-term volatility (10-day) to long-term volatility (30-day) to capture abnormal price fluctuation patterns.",
            "formulation": "\\\\text{RVI} = \\\\frac{\\\\sigma_{\\\\text{short}}}{\\\\sigma_{\\\\text{long}}}, \\\\quad \\\\sigma_{\\\\text{short}} = \\\\text{STD}(\\$close, 10), \\\\quad \\\\sigma_{\\\\text{long}} = \\\\text{STD}(\\$close, 30)",
            "variables": {
                "STD($close, 10)": "10-day rolling standard deviation of closing prices",
                "STD($close, 30)": "30-day rolling standard deviation of closing prices"
            }
        },
        "RVI_5_20": {
            "description": "Alternative Relative Volatility Index with shorter-term (5-day) and longer-term (20-day) volatility comparison for more responsive regime detection.",
            "formulation": "\\\\text{RVI} = \\\\frac{\\\\sigma_{\\\\text{short}}}{\\\\sigma_{\\\\text{long}}}, \\\\quad \\\\sigma_{\\\\text{short}} = \\\\text{STD}(\\$close, 5), \\\\quad \\\\sigma_{\\\\text{long}} = \\\\text{STD}(\\$close, 20)",
            "variables": {
                "STD($close, 5)": "5-day rolling standard deviation of closing prices",
                "STD($close, 20)": "20-day rolling standard deviation of closing prices"
            }
        }
    }
    
    '''

    def escape_latex_for_json(s):
        import re
        # 将单个反斜杠替换为四个反斜杠（适用于 LaTeX 公式在 JSON 中的表示）
        return re.sub(r'\\', r'\\\\', s)


    # escaped_str = escape_latex_for_json(test_str)
    # data = json.loads(escaped_str)
    #
    # data = json.loads(test_str)
    # print(data.keys())

def fix_unescaped_quotes(json_str: str) -> str:
    """
    智能修复 value 中未转义的双引号
    思路：手动扫描字符串，找到 key-value 的 value 部分，在 value 中的非法 " 替换成 \"
    """
    in_string = False
    escape = False
    brace_stack = []
    result = []
    key_mode = True  # 当前是否处于 key 的处理阶段
    quote_count = 0  # 当前 key/value 内部的引号计数

    i = 0
    while i < len(json_str):
        char = json_str[i]

        if char == '"' and not escape:
            in_string = not in_string
            quote_count += 1
            result.append(char)
            i += 1
            continue

        if in_string:
            if char == '\\':
                escape = not escape
            elif char == '"' and not escape:
                # 进入字符串内但没有转义，非法 quote
                result.append('\\"')
                i += 1
                continue
            else:
                escape = False
            result.append(char)
            i += 1
            continue

        if char == ':':
            # 开始进入 value
            key_mode = False
            quote_count = 0
        elif char in '\n\r':
            key_mode = True
        result.append(char)
        i += 1

    return ''.join(result)

import re
import json

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

    # 尝试修复裸引号
    cleaned = fix_unescaped_quotes(cleaned)
    # 移除尾部多余逗号
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    return cleaned


def test_json_parse3():
    test_str = '''
    最后，编写Python代码时需要注意正确的索引处理和分组操作。
    </think>
    
    {
        "code": "import pandas as pd\n\ndef calculate_implied_volatility_rank_10d():\n    # Read data\n    df = pd.read_hdf('daily_pv.h5', key='data')\n    \n    # Calculate daily returns\n    df['daily_return'] = df.groupby(level='instrument')['close'].pct_change()\n    \n    # Calculate 20-day rolling volatility (sigma)\n    df['sigma'] = df.groupby(level='instrument')['daily_return'].rolling(20).std().values\n    \n    # Calculate percentile rank in the last 20 days\n    def calculate_rank(x):\n        return x.rank(pct=True)[-1] * 100\n    \n    df['implied_volatility_rank_10d'] = df.groupby(level='instrument')['sigma'].rolling(20).apply(calculate_rank).values\n    \n    # Save result\n    df[['implied_volatility_rank_10d']].to_hdf('result.h5', key='data')\n\nif __name__ == '__main__':\n    calculate_implied_volatility_rank_10d()"
    }
    '''

    from rdagent.oai.llm_utils import extract_json_objects
    json_result = extract_json_objects(test_str)

    print(json_result)

    import re
    # 提取 code 内容（中间双引号内的部分）
    match = re.search(r'"code": "(.*)"\s*}', test_str, re.DOTALL)
    if match:
        code_raw = match.group(1)

        # 替换字符串中的特殊字符为合法 JSON 字符串格式
        code_fixed = code_raw.replace('\\', '\\\\') \
                             .replace('"', '\\"') \
                             .replace('\n', '\\n')

        # 构造合法 JSON 字符串
        json_fixed_str = '{"code": "' + code_fixed + '"}'

        # 解析为 dict
        data = json.loads(json_fixed_str)
        print(data['code'])  # 输出还原后的代码
    else:
        print("匹配失败")

    import re



    all_response = '''{
        "final_decision": false,
        "final_feedback": "The implementation contains multiple critical errors: 1) Function name 'Volatility-Volume_Composite_Factor_(20D)' is invalid due to special characters and parentheses in the name. 2) The rolling window calculation uses min_periods=1 which violates the 20-day requirement. 3) The volume change calculation incorrectly uses pct_change() instead of absolute value ratio as specified. 4) The output column name does not match the factor name. 5) Missing NaN handling for early period calculations.",
    }'''
    json_target_type = Dict[str, Union[str , bool , int , Dict[str, Union[str , int , float , bool, List[Union[str , int , float , bool]] ]], List[Union[str , int , float , bool , Dict[str, Union[str , int , float , bool ]]]]]]
    TypeAdapter(json_target_type).validate_json(clean_json_str(all_response))
    # TypeAdapter(json_target_type).validate_json(all_response)

def test_json_parse():
    import re

    def clean_control_chars(s: str) -> str:
        # 移除 ASCII 控制字符（\x00-\x1F），保留换行和空格等常用字符
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)


    all_response = r'''{
        "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense\n\ndef calculate_LSTM_Volatility_Prediction():\n    # Read source data\n    df = pd.read_hdf(\"daily_pv.h5\", key=\"data\")\n    \n    # Ensure datetime is sorted for proper calculation\n    df = df.sort_index(level='datetime')\n    \n    # Define window size for historical data (τ=20)\n    window_size = 20\n    \n    # Prepare historical price sequences (open, close, high, low) as input features\n    # Reshape data into 3D tensor: [samples, time_steps, features]\n    X = []\n    for instrument, group in df.groupby(level='instrument'):\n        scaled_data = StandardScaler().fit_transform(group[['\$open', '\$close', '\$high', '\$low']])\n        for i in range(window_size, len(scaled_data)):\n            X.append(scaled_data[i-window_size:i])\n    \n    # Convert to numpy array and reshape\n    X = np.array(X)\n    \n    # Build LSTM model (simplified architecture)\n    model = Sequential([\n        LSTM(50, input_shape=(X.shape[1], X.shape[2])),\n        Dense(1, activation='linear')\n    ])\n    \n    # Compile model\n    model.compile(optimizer='adam', loss='mse')\n    \n    # Predict volatility (simplified - using last available data as input)\n    if len(X) > 0:\n        predictions = model.predict(X).flatten()\n    else:\n        predictions = np.zeros(len(df))\n    \n    # Create result DataFrame with correct structure\n    result_df = pd.DataFrame(predictions, index=df.index[-len(predictions):], columns=['LSTM_Volatility_Prediction'])\n    \n    # Save to HDF5\n    result_df.to_hdf('result.h5', key='data', mode='w')\n\nif __name__ == \"__main__\":\n    calculate_LSTM_Volatility_Prediction()"
    }'''
    json_target_type = dict[str, str ]


    print(clean_control_chars(all_response))

    TypeAdapter(json_target_type).validate_json(clean_control_chars(all_response))
    TypeAdapter(json_target_type).validate_json(clean_json_str(all_response))


def test_json_parse4():
    text_str = '''
    <think>
    好的，我需要根据用户提供的详细信息来生成新的因子。首先，回顾一下用户的场景和需求。用户的目标是基于市场状态分段的动量-波动率交互因子（State-Specific Momentum-Volatility Interaction），解决之前尝试的各种方法中出现的问题，比如过拟合、静态参数适应性差等。

    从之前的假设和反馈来看，用户已经尝试了多种改进方法，包括波动率调整、非线性加权、动态窗口权重等，但效果都不理想。现在需要生成新的因子，结合分段函数策略，通过市场波动率分位数将样本划分为低/中/高波动区间，在每个子区间内独立计算动量因子，并引入波动率敏感度系数动态调节权重。

    首先，我需要确定这个新因子的结构。根据用户提供的Hypothesis，新的因子应该基于市场状态分段，即根据波动率分位数划分不同的市场状态，然后在每个状态下分别计算动量因子，并调整权重。这可能涉及到计算滚动窗口的标准差来确定波动率，然后将其分为不同分位数区间。

    接下来，考虑如何具体实现这个因子。需要定义变量和公式。例如：

    1. **Name**: StateSpecificMomentumVolatilityInteraction
    2. **Description**: 基于市场波动率分位数将样本划分为低/中/高波动区间，在每个子区间内独立计算动量因子，并引入波动率敏感度系数动态调节权重。
    3. **Formulation**: 需要明确公式，可能包括动量计算、波动率分位数划分、加权等步骤。例如，动量在不同波动区间的标准化处理，结合Sigmoid函数调整权重。
    4. **Variables**:
       - $Momentum_{window}$: 不同窗口的动量因子（如10日、20日）
       - $Volatility_{window}$: 波动率计算，比如20日滚动标准差
       - $VolatilityQuantile$: 波动率分位数，将波动率分为低、中、高区间
       - $WeightingFunction$: 使用Sigmoid函数或其他非线性函数动态调整权重

    此外，需要确保代码实现时处理数据对齐问题，正确计算各变量，并保存到HDF5文件。例如，在Python中使用pandas进行分组操作，计算滚动标准差，划分分位数区间，然后根据不同的波动率状态应用不同的动量信号。

    还需要注意用户提到的之前错误，比如数据对齐、缺失值处理等。在代码中需要正确处理这些情况，避免因NaN或无穷大导致的错误。例如，在计算动量时使用ffill或fillna填充缺失值，或者在分位数划分时合理处理边界条件。

    最后，生成JSON格式的输出，包含因子名称、描述、公式和变量说明。确保每个部分都符合用户的要求，并且没有遗漏必要的信息。
    </think>

    {
        "StateSpecificMomentumVolatilityInteraction": {
            "description": "基于市场波动率分位数将样本划分为低/中/高波动区间，在每个子区间内独立计算动量因子，并引入波动率敏感度系数动态调节权重",
            "formulation": "\\\\text{Factor} = \\\\begin{cases}\\n\\\\frac{\\\\text{Momentum}_{10d}}{\\\\sigma_{20d}} & \\\\text{if } \\\\sigma_{20d} \\\\in [Q_0, Q_{33}) \\\\\\\\\\n\\\\frac{\\\\text{Momentum}_{20d}}{\\\\sigma_{20d}} & \\\\text{if } \\\\sigma_{20d} \\\\in [Q_{33}, Q_{66}) \\\\\\\\\\n\\\\frac{\\\\text{Momentum}_{10d} + \\\\text{Momentum}_{20d}}{2\\\\sigma_{20d}} & \\\\text{if } \\\\sigma_{20d} \\\\in [Q_{66}, Q_{100}]\\n\\\\end{cases}",
            "variables": {
                "Momentum_10d": "过去10日收盘价相对变化率",
                "Momentum_20d": "过去20日收盘价相对变化率",
                "Volatility_20d": "20日滚动标准差（波动率）",
                "VolatilityQuantile": "基于分位数的市场状态划分（Q0-Q33/Q66）",
                "SigmoidWeight": "基于波动率的动态权重函数"
            }
        },
        "VolatilityAdjustedMomentumRatio": {
            "description": "结合动量比值与波动率调整的双重过滤机制，通过分位数标准化增强信号鲁棒性",
            "formulation": "\\\\text{Factor} = \\\\frac{\\\\frac{\\\\text{Momentum}_{10d}}{\\\\text{Momentum}_{20d}}}{\\\\Phi^{-1}(\\\\text{VolatilityRank})}",
            "variables": {
                "Momentum_10d": "10日动量因子",
                "Momentum_20d": "20日动量因子",
                "VolatilityRank": "基于分位数的波动率标准化值",
                "PhiInverse": "正态分布分位数函数"
            }
        }
    }
    '''
    from rdagent.oai.llm_utils import extract_json_objects, parse_json_result
    json_result = extract_json_objects(text_str)
    json_str = parse_json_result(text_str)

    print(json_result)
    print('@@@' * 20)
    print(json_str)



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

def test_json_parse5():

    all_response = r'''{
    "final_decision": false,
    "final_feedback": "The code contains multiple syntax errors and logical issues. 1) The use of backslash escape characters in string literals like \"daily_pv.h5\" is invalid and causes a SyntaxError. 2) There's a typo in the loop 'for i in range(len(prices")):' which would cause a syntax error. 3) The code incorrectly uses $close instead of VIX index data for calculating sigma_vix, violating the factor formulation. 4) The Hurst exponent calculation function may not be correctly implemented. These issues prevent the code from generating valid factor values."
}'''
    # from typing import Dict, Any
    # json_target_type = Dict[str, Any],
    #
    # TypeAdapter(json_target_type).validate_json(all_response)


    all_response = clean_json_str(all_response)

    print(all_response)
    from typing import Dict, List, Union

    # 定义嵌套的通用类型
    JsonTargetType = Dict[str, Union[str , bool , int , Dict[str, Union[str , int , float , bool, List[Union[str , int , float , bool]] ]], List[Union[str , int , float , bool , Dict[str, Union[str , int , float , bool ]]]]]]

    # 使用 TypeAdapter
    print( transform_json_string( all_response ) )

    json.loads( all_response )
    # TypeAdapter(JsonTargetType).validate_json( all_response )
    # TypeAdapter(JsonTargetType).validate_json(transform_json_string( all_response ))


