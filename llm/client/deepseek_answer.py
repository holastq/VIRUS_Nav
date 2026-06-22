import os

from llm.utils.get_sys_prompt import get_similar_answer_prompt
from openai import OpenAI


def _build_client():
    # Support OpenAI-compatible endpoints (e.g. Qwen/vLLM) while keeping DeepSeek defaults.
    api_key = (
        os.getenv("QWEN_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or "write your api key here"
    )
    base_url = (
        os.getenv("QWEN_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def deepseek_respond(prompt, model=None):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt
    }
    history = system_prompts + [msg]

    model_name = (
        model
        or os.getenv("QWEN_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )

    response = _build_client().chat.completions.create(
        model=model_name,
        messages=history,
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    deepseek_respond('dining table')