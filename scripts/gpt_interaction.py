import os

from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env')

MODEL_NAME = 'gpt-4'


def query_template(queries: List[str], init_message: str) -> List[Tuple]:
    """
    Method to query OpenAI model.
    :param queries: list of texts to query.
    :param init_message: instructions to model before main queries.
    :return: list of answers, including answer to `init_message` if provided.
    """
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    responses = []

    for query in queries:
        messages = [
            {"role": "system", "content": init_message},
            {"role": "user", "content": query}
        ]

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            n=1,  # number of generated answers per query,
            temperature=0  # fully deterministic results, no random creativity
        )

        response = completion.choices[0].message
        responses.append((query, response))

    return responses
