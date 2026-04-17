import json
import time
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import LLMSettings
from .provider_capabilities import resolve_provider_capabilities


class ThinkingChatOpenAI(ChatOpenAI):
    """
    Extended ChatOpenAI that supports thinking/reasoning mode for compatible models.
    Supports models like DeepSeek-R1, Qwen-QwQ, etc.
    """
    
    thinking_enabled: bool = False
    thinking_budget: int = 8192
    thinking_transport: str = "auto"
    show_thinking: bool = True
    
    def __init__(
        self,
        thinking_enabled: bool = False,
        thinking_budget: int = 8192,
        thinking_transport: str = "auto",
        show_thinking: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        if thinking_transport == "auto":
            capabilities = resolve_provider_capabilities(
                model_name=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
            )
            self.thinking_transport = capabilities.thinking_transport
        else:
            self.thinking_transport = thinking_transport
        self.show_thinking = show_thinking
    
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Override to inject thinking parameters into API request."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        stream_requested = kwargs.get("stream") is True
        response_format = kwargs.get("response_format", payload.get("response_format"))
        if isinstance(response_format, dict):
            response_format_type = response_format.get("type")
        elif response_format is not None:
            response_format_type = getattr(response_format, "type", response_format)
        else:
            response_format_type = response_format
        json_mode = response_format_type in {"json_object", "json_schema"} or (
            isinstance(response_format, dict)
            and ("json_schema" in response_format or str(response_format_type or "").startswith("json"))
        )
        if not json_mode and response_format is not None:
            json_mode = True
        if not json_mode and ("response_format" in payload or "response_format" in kwargs):
            json_mode = True
        if not json_mode and ("json_schema" in kwargs or "json_schema" in payload):
            json_mode = True

        if self.thinking_enabled and self.thinking_transport == "extra_body":
            if "extra_body" not in payload:
                payload["extra_body"] = {}

            if json_mode:
                payload["extra_body"].pop("enable_thinking", None)
                payload["extra_body"].pop("thinking_budget", None)
                if not payload["extra_body"]:
                    payload.pop("extra_body", None)
            elif stream_requested:
                payload["extra_body"]["enable_thinking"] = True
                payload["extra_body"]["thinking_budget"] = self.thinking_budget

                if "stream_options" not in payload:
                    payload["stream_options"] = {}
                payload["stream_options"]["include_usage"] = True
            else:
                payload["extra_body"]["enable_thinking"] = False
                payload["extra_body"].pop("thinking_budget", None)
        
        return payload


class LLMService:
    """Abstraction over model creation to keep graph definition decoupled."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def create_chat_model(self) -> BaseChatModel:
        if self.settings.mode == "API":
            api_base = (self.settings.api_base or "").strip().lower()
            is_local_api = api_base.startswith("http://localhost") or api_base.startswith("http://127.0.0.1") or api_base.startswith("http://0.0.0.0") or api_base.startswith("https://localhost") or api_base.startswith("https://127.0.0.1") or api_base.startswith("https://0.0.0.0")
            if not (self.settings.api_key or "").strip() and not is_local_api:
                raise ValueError(
                    "LLM_MODE=API 但未配置 LLM_API_KEY。请在 .env 中设置 LLM_API_KEY，或将 LLM_MODE=Local 使用本地模型。"
                )
            return create_compatible_chat_openai(
                model=self.settings.model,
                api_key=self.settings.api_key,
                base_url=self.settings.api_base,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                streaming=self.settings.streaming,
                max_retries=3,
                provider_hint=self.settings.provider,
                thinking_enabled=self.settings.thinking_enabled,
                thinking_budget=self.settings.thinking_budget,
                show_thinking=self.settings.show_thinking,
            )

        if self.settings.mode == "Local":
            backend = self.settings.local_backend
            if backend == "Auto":
                model_path_lower = (self.settings.local_model_path or "").lower()
                backend = "VLLM" if "fp8" in model_path_lower else "HF"
            if backend == "VLLM":
                from .local_hf_chat import LocalVLLMChatModel
                return LocalVLLMChatModel(
                    self.settings.local_model_path,
                    max_new_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    concise_mode=self.settings.local_concise_mode,
                    repetition_penalty=self.settings.local_repetition_penalty,
                    dtype=self.settings.local_vllm_dtype,
                    tensor_parallel_size=self.settings.local_vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.settings.local_vllm_gpu_memory_utilization,
                    max_model_len=self.settings.local_vllm_max_model_len,
                )
            from .local_hf_chat import LocalHFChatModel
            return LocalHFChatModel(
                self.settings.local_model_path,
                max_new_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                concise_mode=self.settings.local_concise_mode,
                repetition_penalty=self.settings.local_repetition_penalty,
            )

        msg = f"Unsupported LLM_MODE={self.settings.mode}. Use API or Local."
        raise NotImplementedError(msg)

    @staticmethod
    def _patch_str_response(model: ChatOpenAI, provider_hint: str = "") -> ChatOpenAI:
        """
        Some third-party OpenAI-compatible APIs return raw strings instead of
        the usual dict/Typed objects. Patch the model to wrap such responses
        into an OpenAI-like payload so LangChain can parse it.
        """

        base_create = model._create_chat_result
        base_payload = model._get_request_payload

        def _safe_create(self, response, generation_info=None, *args, **kwargs):
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except Exception:
                    response = {
                        "id": "patched-str-response",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": getattr(self, "model_name", ""),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": response},
                                "finish_reason": "stop",
                            }
                        ],
                    }

            if generation_info is None and args:
                generation_info = args[0]
            if generation_info is None and "generation_info" in kwargs:
                generation_info = kwargs.get("generation_info")
            if generation_info is None:
                return base_create(response)
            return base_create(response, generation_info)

        def _safe_payload(self, input_, *, stop=None, **kwargs):
            payload = base_payload(input_, stop=stop, **kwargs)
            response_format = kwargs.get("response_format", payload.get("response_format"))
            if isinstance(response_format, dict):
                response_format_type = response_format.get("type")
            elif response_format is not None:
                response_format_type = getattr(response_format, "type", response_format)
            else:
                response_format_type = response_format
            json_mode = response_format_type in {"json_object", "json_schema"} or (
                isinstance(response_format, dict)
                and ("json_schema" in response_format or str(response_format_type or "").startswith("json"))
            )
            if not json_mode and response_format is not None:
                json_mode = True
            if not json_mode and ("response_format" in payload or "response_format" in kwargs):
                json_mode = True
            if not json_mode and ("json_schema" in kwargs or "json_schema" in payload):
                json_mode = True
            capabilities = resolve_provider_capabilities(
                model_name=str(getattr(self, "model_name", "") or getattr(self, "model", "") or ""),
                base_url=str(getattr(self, "openai_api_base", "") or getattr(self, "base_url", "") or ""),
                provider_hint=provider_hint,
            )
            if not capabilities.supports_system_messages and isinstance(payload.get("messages"), list):
                payload["messages"] = [
                    {**message, "role": "user"}
                    if isinstance(message, dict) and message.get("role") == "system"
                    else message
                    for message in payload["messages"]
                ]
            if json_mode:
                extra_body = payload.get("extra_body", {})
                extra_body["enable_thinking"] = False
                extra_body.pop("thinking_budget", None)
                if extra_body:
                    payload["extra_body"] = extra_body
                else:
                    payload.pop("extra_body", None)
            return payload

        model._create_chat_result = _safe_create.__get__(model, ChatOpenAI)
        model._get_request_payload = _safe_payload.__get__(model, ChatOpenAI)
        return model


def create_compatible_chat_openai(
    *,
    provider_hint: str = "",
    thinking_enabled: bool = False,
    thinking_budget: int = 8192,
    thinking_transport: str = "auto",
    show_thinking: bool = True,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client with provider capability patching."""

    capabilities = resolve_provider_capabilities(
        model_name=str(kwargs.get("model", "") or ""),
        base_url=str(kwargs.get("base_url", "") or ""),
        provider_hint=provider_hint,
    )
    resolved_provider_hint = provider_hint or capabilities.provider

    if thinking_enabled and capabilities.supports_thinking:
        llm = ThinkingChatOpenAI(
            thinking_enabled=True,
            thinking_budget=thinking_budget,
            thinking_transport=thinking_transport or capabilities.thinking_transport,
            show_thinking=show_thinking,
            **kwargs,
        )
    else:
        llm = ChatOpenAI(**kwargs)

    return LLMService._patch_str_response(llm, provider_hint=resolved_provider_hint)
