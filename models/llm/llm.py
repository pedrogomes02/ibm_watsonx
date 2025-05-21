import logging
import requests
import time
import json

from collections.abc import Generator
from typing import Optional, Union
from decimal import Decimal

from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    AssistantPromptMessage,
)

logger = logging.getLogger(__name__)


class IbmWatsonxLargeLanguageModel(LargeLanguageModel):
    """
    Model class for ibm_watsonx large language model.
    """

    def _get_access_token(self, api_key: str) -> str:
        try:
            logging.warning("Fetching IBM access token...")
            response = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": api_key
                }
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            logger.error("Failed to fetch IBM access token: %s", str(e))
            raise CredentialsValidateFailedError("Invalid IBM API Key or network error.")

    def serialize_message(self, msg):
        base = {
            "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            base["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls
            ]
        else:
            base["content"] = msg.content
        if hasattr(msg, "tool_call_id"):
            base["tool_call_id"] = msg.tool_call_id
        return base

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """

        api_key = credentials.get("ibm_api_key")
        project_id = credentials.get("ibm_project_id")
        ibm_url = credentials.get("ibm_url")
        if not api_key or not project_id:
            raise CredentialsValidateFailedError("Missing required credentials.")

        token = self._get_access_token(api_key)
        messages_payload = [self.serialize_message(m) for m in prompt_messages]

        request_payload = {
            "messages": messages_payload,
            "temperature": model_parameters.get("temperature", 0.7),
            "max_tokens": model_parameters.get("max_tokens", 1024),
            "top_p": model_parameters.get("top_p", 1),
            "frequency_penalty": model_parameters.get("frequency_penalty", 0),
            "presence_penalty": model_parameters.get("presence_penalty", 0),
            "stop": stop,
            "model_id": model,
            "project_id": project_id
        }

        if tools:
            tools_payload = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }} for tool in tools
            ]
            request_payload["tools"] = tools_payload

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if stream:
            url = f"{ibm_url}/ml/v1/text/chat_stream?version=2023-05-29"
        else:
            url = f"{ibm_url}/ml/v1/text/chat?version=2023-05-29"

        try:
            start_time = time.time()
            response = requests.post(
                url, headers=headers, json=request_payload, stream=stream, timeout=(10, 300)
            )
            response.raise_for_status()

            if stream:
                def stream_generator():
                    full_text = ""
                    index = 0

                    for line in response.iter_lines(decode_unicode=True):
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        try:
                            payload = json.loads(line[len("data:"):].strip())
                            choices = payload.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            delta_text = delta.get("content", "")
                            tool_calls = delta.get("tool_calls", [])

                            if delta_text:
                                full_text += delta_text
                                yield LLMResultChunk(
                                    model=model,
                                    prompt_messages=prompt_messages,
                                    delta=LLMResultChunkDelta(
                                        index=index,
                                        message=AssistantPromptMessage(content=delta_text, role="assistant")
                                    )
                                )
                                index += 1

                            elif tool_calls:
                                for tool_call in tool_calls:
                                    yield LLMResultChunk(
                                        model=model,
                                        prompt_messages=prompt_messages,
                                        delta=LLMResultChunkDelta(
                                            index=index,
                                            message=AssistantPromptMessage(
                                                tool_calls= tool_calls,
                                                role="assistant"
                                            )
                                        )
                                    )
                                    index += 1

                            if finish_reason == "stop":
                                break

                        except json.JSONDecodeError as e:
                            logger.warning("Skipping malformed data line: %r (%s)", line, e)
                            continue

                return stream_generator()
            else:
                data = response.json()
                logger .warning("Response data: %s", data)
                choice = data.get("choices", [{}])[0]
                message_data = choice.get("message", {})
                tool_calls = message_data.get("tool_calls") or []
                content = message_data.get("content", "")

                usage_data = data.get("usage", {})
                latency = time.time() - start_time

                assistant_message = AssistantPromptMessage(content=content,tool_calls=tool_calls, role="assistant")

                return LLMResult(
                    model=model,
                    prompt_messages=prompt_messages,
                    message=assistant_message,
                    usage={
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "prompt_unit_price": Decimal("0"),
                        "prompt_price_unit": Decimal("0"),
                        "prompt_price": Decimal("0.0"),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "completion_unit_price": Decimal("0"),
                        "completion_price_unit": Decimal("0"),
                        "completion_price": Decimal("0.0"),
                        "total_tokens": usage_data.get("total_tokens", 0),
                        "total_price": Decimal("0.0"),
                        "currency": "USD",
                        "latency": latency,
                    },
                )
  
        except Exception as e:
            logger.error("LLM invocation failed: %s", str(e))
            raise RuntimeError("LLM call failed")

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        logging.warning("Getting number of tokens...")
        text = ""
        for msg in prompt_messages:
            if isinstance(msg.content, str):
                text += msg.content
            elif isinstance(msg.content, list):
                for content_piece in msg.content:
                    if hasattr(content_piece, 'data'):
                        text += str(content_piece.data)
        return len(text.split())

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            logging.warning("Validating credentials...")
            api_key = credentials.get("ibm_api_key")
            if not api_key:
                raise CredentialsValidateFailedError("Missing API key.")

            self._get_access_token(api_key)
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema of the base model
        but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
            parameter_rules=[],
        )
        logging.warning("Customizable model schema: %s", entity)

        return entity
    
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        logging.warning("Mapping invoke errors...")
        return {
            Exception: [],
            InvokeAuthorizationError: [
                requests.exceptions.InvalidHeader,  # Missing or Invalid API Key
            ],
            InvokeBadRequestError: [
                requests.exceptions.HTTPError,  # Invalid Endpoint URL or model name
                requests.exceptions.InvalidURL,  # Misconfigured request or other API error
            ],
            InvokeRateLimitError: [
                requests.exceptions.RetryError  # Too many requests sent in a short period of time
            ],
            InvokeServerUnavailableError: [
                requests.exceptions.ConnectionError,  # Engine Overloaded
                requests.exceptions.HTTPError,  # Server Error
            ],
            InvokeConnectionError: [
                requests.exceptions.ConnectTimeout,  # Timeout
                requests.exceptions.ReadTimeout,  # Timeout
            ],
        }

