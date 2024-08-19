import asyncio
import os

from openai import AsyncOpenAI
from tqdm import tqdm

EMAIL_TOKEN = "email"
URL_TOKEN = "url"


openai = AsyncOpenAI(
    api_key=os.environ["api_key"],
    base_url="https://api.deepinfra.com/v1/openai",
)


async def query_server_in_chunk(
    chat_messages,
    model_name,
    chunk_size=10,
):
    async def _predict(messages, model_name):
        chat_completion = await openai.chat.completions.create(
            model=model_name, messages=messages, temperature=0.1, max_tokens=8192
        )
        return chat_completion.choices[0].message.content

    generated_list = []
    chunk_original_sentences = []
    for content in tqdm((chat_messages), total=len(chat_messages)):

        chunk_original_sentences.append(content)
        if len(chunk_original_sentences) >= chunk_size:
            response_list = await asyncio.gather(
                *[
                    _predict(current_content, model_name=model_name)
                    for current_content in chunk_original_sentences
                ]
            )
            generated_list.extend(
                [repr(response.replace("\n", " ")) for response in response_list]
            )
            chunk_original_sentences = []

    if len(chunk_original_sentences) > 0:
        response_list = await asyncio.gather(
            *[
                _predict(current_content, model_name=model_name)
                for current_content in chunk_original_sentences
            ]
        )
        generated_list.extend(
            [repr(response.replace("\n", " ")) for response in response_list]
        )

    return generated_list
