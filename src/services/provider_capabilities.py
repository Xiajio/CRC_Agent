from dataclasses import dataclass, replace
from typing import Literal


StructuredOutputStrategy = Literal["auto", "raw_first", "structured_first"]
ThinkingTransport = Literal["none", "extra_body"]


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    supports_system_messages: bool = True
    structured_output_strategy: StructuredOutputStrategy = "auto"
    thinking_transport: ThinkingTransport = "none"

    @property
    def supports_thinking(self) -> bool:
        return self.thinking_transport != "none"


_BASE_PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(provider="openai"),
    "openai_compatible": ProviderCapabilities(provider="openai_compatible"),
    "minimax": ProviderCapabilities(
        provider="minimax",
        supports_system_messages=False,
        structured_output_strategy="raw_first",
    ),
    "deepseek": ProviderCapabilities(provider="deepseek"),
    "qwen": ProviderCapabilities(provider="qwen"),
}


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_provider_hint(provider_hint: str | None) -> str:
    hint = _normalize_text(provider_hint)
    if hint in {"dashscope", "tongyi", "aliyun"}:
        return "qwen"
    if hint in _BASE_PROVIDER_CAPABILITIES:
        return hint
    return ""


def resolve_provider_name(
    model_name: str | None = None,
    base_url: str | None = None,
    provider_hint: str | None = None,
) -> str:
    normalized_hint = _normalize_provider_hint(provider_hint)
    if normalized_hint:
        return normalized_hint

    model_text = _normalize_text(model_name)
    base_url_text = _normalize_text(base_url)
    combined = f"{model_text} {base_url_text}"

    if "minimax" in combined or "minimaxi" in combined:
        return "minimax"
    if "deepseek" in combined:
        return "deepseek"
    if "qwen" in combined or "qwq" in combined or "dashscope" in combined:
        return "qwen"
    if "openai.com" in combined or "gpt-" in model_text or model_text.startswith(("o1", "o3", "o4")):
        return "openai"
    return "openai_compatible"


def _supports_extra_body_thinking(provider: str, model_name: str | None = None) -> bool:
    model_text = _normalize_text(model_name)
    if provider == "deepseek":
        return "deepseek-r1" in model_text or model_text.endswith("-r1") or "reasoner" in model_text
    if provider == "qwen":
        return "qwq" in model_text
    return False


def resolve_provider_capabilities(
    model_name: str | None = None,
    base_url: str | None = None,
    provider_hint: str | None = None,
) -> ProviderCapabilities:
    provider = resolve_provider_name(model_name=model_name, base_url=base_url, provider_hint=provider_hint)
    base_caps = _BASE_PROVIDER_CAPABILITIES.get(provider, _BASE_PROVIDER_CAPABILITIES["openai_compatible"])
    if _supports_extra_body_thinking(provider, model_name=model_name):
        return replace(base_caps, thinking_transport="extra_body")
    return base_caps
