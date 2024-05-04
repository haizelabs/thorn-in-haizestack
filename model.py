import anthropic
import openai
import cohere
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Optional, Union


class ModelClient:

    def __init__(
        self,
        client: Union[cohere.Client, openai.OpenAI, anthropic.Anthropic],
        model_name: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, msgs: list[dict[str, str]], **kwargs):

        match type(self.client):
            case anthropic.Anthropic:
                response = self.client.messages.create(
                    system=self.system_prompt,
                    messages=msgs,
                    model=self.model_name,
                    max_tokens=4096,
                    **kwargs,
                )
                response = response.content[-1].text
            case cohere.Client:
                response = self.client.chat(
                    chat_history=msgs[:-1],
                    message=msgs[-1]["message"],
                    model=self.model_name,
                    **kwargs,
                )
                response = response.chat_history[-1].message
            case openai.OpenAI:
                response = self.client.chat.completions.create(
                    messages=msgs,
                    model=self.model_name,
                    **kwargs,
                )
                response = response.choices[0].message.content

        return response

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def tokenize(self, text: str) -> list[int]:
        match type(self.client):
            case anthropic.Anthropic:
                return self.client.get_tokenizer().encode(text).ids
            case cohere.Client:
                return self.client.tokenize(text=text, model=self.model_name).tokens
            case openai.OpenAI:
                enc = tiktoken.encoding_for_model(self.model_name)
                return enc.encode(text)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def detokenize(self, tokens: list[int]) -> str:
        match type(self.client):
            case anthropic.Anthropic:
                return self.client.get_tokenizer().decode(tokens)
            case cohere.Client:
                return self.client.detokenize(tokens=tokens, model=self.model_name).text
            case openai.OpenAI:
                enc = tiktoken.encoding_for_model(self.model_name)
                return enc.decode(tokens)


def format_question(
    mc: ModelClient, retrieval_question: str, context: str
) -> list[dict[str, str]]:
    match type(mc.client):
        case anthropic.Anthropic:
            return [
                {
                    "role": "user",
                    "content": f" <context>{context}</context>\n\n{retrieval_question} Don't give information outside the document or repeat your findings",
                },
            ]

        case cohere.Client:
            return [
                {
                    "role": "System",
                    "message": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
                },
                {"role": "User", "message": context},
                {
                    "role": "User",
                    "message": f"{retrieval_question} Don't give information outside the document or repeat your findings",
                },
            ]

        case openai.OpenAI:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
                },
                {"role": "user", "content": context},
                {
                    "role": "user",
                    "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
                },
            ]
