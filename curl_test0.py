
from litellm import completion, embedding


response = completion(
    model="lm_studio/qwen3-30b-a3b",  # 模型名称需与你部署或服务上的一致
    messages=[
        {"role": "user", "content": "写一首关于春天的现代诗"}
    ],
    api_base="http://127.0.0.1:4000",  # 替换为你的 API 地址
    api_key="sk-1234"  # 本地部署的 API 通常不需要 key，可以留空
)

print(response["choices"][0]["message"]["content"])




input_contents = ["factor_name: VolumeWeightedPriceMomentum_10d\nfactor_description: Volume-weighted price momentum captures trend strength by weighting price changes with corresponding trading volumes over a 10-day window.\nfactor_formulation: \\text{VolumeWeightedPriceMomentum}_{t} = \\frac{\\sum_{i=t-9}^{t} (\\Delta P_i \\times V_i)}{\\sum_{i=t-9}^{t} V_i}\nvariables: {'price_change': 'Daily price change calculated as $close(t) - $close(t-1)', 'volume': '$volume: Trading volume on the given day'}"]

response = embedding(
    model="lm_studio/text-embedding-bge-m3",  # 模型名称需与你部署或服务上的一致
    input=input_contents,
    api_base="http://127.0.0.1:4000",  # 替换为你的 API 地址
    api_key="sk-1234"  # 本地部署的 API 通常不需要 key，可以留空
)

print(response)
