import asyncio
import os
import time
import logging
from openai import AsyncOpenAI, RateLimitError
from tqdm import tqdm


logger = logging.getLogger(__name__)
EMAIL_TOKEN = "email"
URL_TOKEN = "url"


default_openai_client = AsyncOpenAI(
    api_key=os.getenv("api_key", ""),
    base_url="https://api.deepinfra.com/v1/openai",
)


async def query_server_in_chunk(
    chat_messages,
    model_name,
    chunk_size=10,
    openai_client=default_openai_client
):
    total_model_time = 0
    async def _predict(messages, model_name):
        chat_completion = await openai_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.1, max_tokens=8192
        )
        return chat_completion.choices[0].message.content

    generated_list = []
    chunk_original_sentences = []
    for content in tqdm((chat_messages), total=len(chat_messages)):
        chunk_original_sentences.append(content)
        if len(chunk_original_sentences) >= chunk_size:
            try:
                start_time = time.time()
                response_list = await asyncio.gather(
                    *[
                        _predict(current_content, model_name=model_name)
                        for current_content in chunk_original_sentences
                    ]
                )
                total_model_time += time.time() - start_time
            except RateLimitError:
                time.sleep(30)
                start_time = time.time()
                response_list = await asyncio.gather(
                    *[
                        _predict(current_content, model_name=model_name)
                        for current_content in chunk_original_sentences
                    ]
                )
                total_model_time += time.time() - start_time
            generated_list.extend(
                [repr(response.replace("\n", " ")) for response in response_list]
            )
            chunk_original_sentences = []

    if len(chunk_original_sentences) > 0:
        try:
            start_time = time.time()
            response_list = await asyncio.gather(
                *[
                    _predict(current_content, model_name=model_name)
                    for current_content in chunk_original_sentences
                ]
            )
            total_model_time += time.time() - start_time
        except RateLimitError:
            time.sleep(30)
            start_time = time.time()
            response_list = await asyncio.gather(
                *[
                    _predict(current_content, model_name=model_name)
                    for current_content in chunk_original_sentences
                ]
            )
            total_model_time += time.time() - start_time
        generated_list.extend(
            [repr(response.replace("\n", " ")) for response in response_list]
        )
    
    print(f"total model inference time: {round(total_model_time, 2)}")
    return generated_list
