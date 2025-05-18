'''
import requests

url = "http://127.0.0.1:1234/v1/chat/completions"
url = "http://127.0.0.1:4000/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json"
}
payload = {
    "model": "lm_studio/qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "hi"}]
}
response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.text)
'''


'''
from litellm import completion

response = completion(
    model="lm_studio/qwen3-30b-a3b",  # 模型名称需与你部署或服务上的一致
    messages=[
        {"role": "user", "content": "写一首关于春天的现代诗"}
    ],
    api_base="http://127.0.0.1:4000",  # 替换为你的 API 地址
    api_key="sk-1234"  # 本地部署的 API 通常不需要 key，可以留空
)

print(response["choices"][0]["message"]["content"])
'''

import re
import json

text = 'some log text { "name": "Alice", "age": 30, "city": "Beijing" } more text'

text = '''I think that's it. Let me put this into the required JSON format.
</think>

{
    "volume_adjusted_price_momentum": {
        "description": "Captures price momentum adjusted by volume dynamics to reflect stronger trend conviction through normalized price changes.",
        "formulation": "\\text{Volume-Adjusted Momentum} = \\frac{\\frac{\\text{Close}_t - \\text{Close}_{t-5}}{\\text{Close}_{t-5}}}{\\text{RollingMean}(\\text{Volume}, 10)}",
        "variables": {
            "Close": "Daily closing price of the instrument used to calculate 5-day price change.",
            "Volume": "Daily trading volume of the instrument used to compute 10-day rolling average for normalization."
        }
    }
}
'''

import regex
import json

import regex
import json
import ast
import regex
import json
import json
import regex  # 更强大的正则库，比 re 更适合处理嵌套结构

def repair_and_parse_json(text: str):
    # 尝试用正则提取出 JSON 字符串块
    match = re.search(r'({\s*"code"\s*:\s*")(.*)("})', text, re.DOTALL)
    # re.search(r'([^"]+)"\s*:', text, re.DOTALL).groups() ('code')
    if not match:
        raise ValueError("未找到包含 code 的 JSON 对象")

    prefix, code_body, suffix = match.groups()

    # 对中间的 Python 代码部分进行 JSON 字符串转义
    escaped_code = json.dumps(code_body)[1:-1]  # 用 json.dumps 自动转义，再去除首尾引号

    # 构造修复后的 JSON 字符串
    repaired_json = prefix + escaped_code + suffix

    # 加载为 Python 对象
    return json.loads(repaired_json)

import json
import re
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

                    try:
                        parsed = repair_and_parse_json(raw_json)
                        results.append(parsed)
                    except:
                        pass

                    start_index = None

    return results

text11 = '''
log: [{"a": 1, "b": [1, 2, {"x": 3}]}, {"a":23}]
data: [1, 2, 3, {"nested": "yes"}]
corrupted: {invalid json here}
'''

text = r'''
</think>

{
    "Volume-Price Momentum Convergence (20-day)": {
        "description": "Measures the relative strength of volume changes versus price changes over a rolling window to capture liquidity-driven market sentiment. It quantifies the divergence between volume and price trends, identifying assets where liquidity dynamics signal upcoming reversion or continuation patterns.",
        "formulation": "\\text{Factor} = \\frac{\\left( \\frac{V_t}{V_{t-20}} - 1 \\right) - \\left( \\frac{P_t}{P_{t-20}} - 1 \\right)}{\\sigma_D}",
        "variables": {
            "$volume": "The volume data of the instrument.",
            "$close": "The closing price data of the instrument."
        }
    }
}
'''


text = r'''

</think>

{
    "code": "import pandas as pd\nimport numpy as np\n\ndef calculate_Volume_Price_Momentum_Ratio_20_Day():\n    df = pd.read_hdf("daily_pv.h5", key="data")\n    df_reset = df.reset_index()\n    df_reset = df_reset.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)\n    \n    # Calculate 20-day volume momentum\n    df_reset['volume_mom'] = df_reset.groupby('instrument')['$volume'].transform(lambda x: (x - x.shift(20)) / x.shift(20))\n    # Calculate 20-day price momentum\n    df_reset['price_mom'] = df_reset.groupby('instrument')['$close'].transform(lambda x: (x - x.shift(20)) / x.shift(20))\n    \n    # Compute the ratio of volume momentum to price momentum\n    df_reset['Volume_Price_Momentum_Ratio_20-Day'] = df_reset['volume_mom'] / df_reset['price_mom']\n    \n    result_df = df_reset.set_index(['datetime', 'instrument'])[['Volume_Price_Momentum_Ratio_20-Day']]\n    result_df.to_hdf("result.h5", key="data")\n\nif __name__ == "__main__":\n    calculate_Volume_Price_Momentum_Ratio_20_Day()"
}
'''

text = '''

Also, check that the factor has a clear formulation with hyperparameters like window size (20 days). The variables would include the close prices for calculating returns and the rolling standard deviation.

I think that's a solid hypothesis. It's simple, uses existing data, and aligns with financial theory. Let me put this into the required JSON structure.
</think>

{
  "hypothesis": "Historical Volatility (20-day Rolling Standard Deviation of Daily Returns)",
  "reason": "This hypothesis introduces a volatility-based factor that captures the dispersion of daily price movements over a defined window. Volatility is a critical risk metric in finance, and its inclusion aligns with the principle that assets with higher volatility may exhibit greater return variability. By focusing on recent historical volatility, the factor aims to quantify market uncertainty, which can influence investor behavior and asset pricing. The 20-day window balances responsiveness to short-term shocks with stability against noise, making it a practical candidate for predictive modeling.",
  "concise_reason": "Volatility quantifies price dispersion, a key risk driver. Short-term volatility captures dynamic market conditions.",
  "concise_observation": "Daily return variability is a persistent feature of financial markets, often linked to regime shifts and liquidity changes.",
  "concise_justification": "If historical volatility reflects underlying market stress, then it can explain asset return anomalies during periods of heightened uncertainty.",
  "concise_knowledge": "If historical volatility is captured through rolling standard deviation of daily returns over 20 days, then it can explain variations in asset price movements driven by changing risk appetite."
}

'''

text = '''
So the final code is as above.
</think>

{
    "code": "import pandas as pd\nimport numpy as np\n\ndef calculate_10_day_return_momentum():\n    # Read the data\n    df = pd.read_hdf("daily_pv.h5", key="data")\n    \n    # Reset index to get 'datetime' and 'instrument' as columns\n    df = df.reset_index()\n    df.columns = ['datetime', 'instrument', '$open', '$close', '$high', '$low', '$volume', '$factor']\n    \n    # Sort the data by instrument and datetime to ensure correct shifting\n    df = df.sort_values(by=['instrument', 'datetime'])\n    \n    # Calculate shifted close values for each day (i=1 to 10)\n    for i in range(1, 11):\n        shift_col_name = f'close_shift_{i}'\n        df[shift_col_name] = df.groupby('instrument')['$close'].shift(i)\n    \n    # Calculate the returns for each of the 10 days and sum them\n    momentum_columns = []\n    for i in range(1, 11):\n        shift_col_name = f'close_shift_{i}'\n        return_col_name = f'return_{i}'\n        df[return_col_name] = (df['$close'] / df[shift_col_name] - 1)\n        momentum_columns.append(return_col_name)\n    \n    # Sum all the returns to get the 10-day momentum\n    df['10_day_momentum'] = df[momentum_columns].sum(axis=1)\n    \n    # Prepare the result DataFrame with MultiIndex and the factor column\n    result_df = df.set_index(['datetime', 'instrument'])[['10_day_momentum']]\n    result_df.columns = ['10-Day Return Momentum']\n    \n    # Save to HDF5 file\n    result_df.to_hdf('result.h5', key='data', mode='w')\n\nif __name__ == "__main__":\n    calculate_10_day_return_momentum()"
}
'''

# text = repr(text)
# jsons = extract_json_objects1(text)
# jsons = extract_json_objects_flexible(text)
jsons = extract_json_objects(text)
# print(jsons)
for j in jsons:
    # print(json.dumps(j, indent=2))
    print(json.dumps(j))
    print(j, type(j), str(j))

