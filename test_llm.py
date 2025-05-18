import litellm
'''
# 加载配置文件
# litellm.load_config("/Users/dabai/work/data/agent_qlib/litellm_lm/config.yaml")
litellm._turn_on_debug()
litellm.api_key = 'http://localhost:4000'
# 打印配置信息，检查 API 密钥是否正确
print(f"API Key: {litellm.api_key}")

from litellm import embedding
import os

os.environ['LM_STUDIO_API_BASE'] = "http://localhost:1234/v1"
response = embedding(
    # model="lm_studio/text-embedding-bge-m3",
    model="lm_studio/text-embedding-bge-m3",
    input=["Hello world"],
)
print(response)

'''
import litellm
litellm.api_key = "sk-1234"
litellm.drop_params=True
import openai
client = openai.OpenAI(
    api_key='sk-1234',             # pass litellm proxy key, if you're using virtual keys -- sk-1234
    base_url="http://0.0.0.0:4000" # litellm-proxy-base url
)

response = client.embeddings.create(model="text-embedding-bge-m3",
                                    extra_headers={
                                        # "Content-Type": "application/json",
                                    },
                                    input="Hello world")
# response = client.embeddings.create(model="lm_studio/text-embedding-bge-m3", input="Hello world")

print('==='*20, response)

response = client.chat.completions.create(
    # model="lm_studio/qwen3-30b-a3b",
    model="qwen3-30b-a3b",
    extra_headers={
        # "Content-Type": "application/json",
    },
    messages = [
        {
            "role": "user",
            "content": "what llm are you"
        }
    ],
)

print('**'*20, response)

