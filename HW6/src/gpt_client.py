# src/gpt_client.py

from openai import OpenAI
from src.config import OPENAI_API_KEY

# 没 key 就直接报错，避免你跑到一半才发现
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env file.")

# 初始化客户端（openai>=1.x 的写法）
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_report(prompt: str, model: str = "gpt-4o") -> str:
    """
    调用 OpenAI 的对话模型，返回纯文本内容。
    prompt: 要发给 GPT 的提示词
    model: 你们课程要用的模型名，默认 gpt-4o，老师如果说用 gpt-4 就改这里
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    # 取第一条回复的文本
    return resp.choices[0].message.content.strip()
