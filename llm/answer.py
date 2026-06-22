from llm.client.deepseek_answer import deepseek_respond
from llm.utils.only_answer import only_answer
from llm.client.ollama_answer import ollama_respond


def get_answer(client, prompt=None):
    if client.llm_client == 'deepseek':
        model = getattr(client, 'openai_model', None)
        respond = deepseek_respond(prompt=prompt, model=model)
    elif client.llm_client == 'ollama':
        respond = ollama_respond(model=client.ollama, prompt=prompt)
    else:
        raise ValueError(
            f"Unsupported llm_client '{client.llm_client}', expected one of: deepseek, ollama"
        )

    similar_answer = only_answer(respond)
    if similar_answer is None:
        raise ValueError(f"Failed to parse LLM answer from response: {respond}")

    return similar_answer, respond