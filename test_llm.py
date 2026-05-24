from openai import OpenAI

client = OpenAI(
    api_key=,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=True,                        # ← 改为 True
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

# 流式处理
for chunk in response:
    delta = chunk.choices[0].delta
    # 通用内容（日常回复）打字输出
    if delta.content:
        print(delta.content, end='', flush=True)
    # 若需要输出推理过程（思考链），可添加：
    # if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
    #     print(delta.reasoning_content, end='', flush=True)