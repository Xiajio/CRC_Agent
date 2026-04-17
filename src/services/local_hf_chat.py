"""
Minimal HuggingFace chat model wrapper for local inference.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional
from threading import Thread

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core import messages as lc_messages
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def _lc_to_hf_role(msg: BaseMessage) -> str:
    """Map LangChain message types to HF chat template roles."""
    role = getattr(msg, "type", "") or ""
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    if role == "system":
        return "system"
    # Tool/other -> flatten as assistant text
    return "assistant"


class LocalHFChatModel(BaseChatModel):
    """
    Lightweight wrapper around a local HF causal LM to mimic a chat model.

    - Uses tokenizer.apply_chat_template to build prompts.
    - Supports optional streaming via TextIteratorStreamer when stream=True.
    - Supports concise_mode to reduce verbose output.
    """

    model_name: str = "local-hf"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        concise_mode: bool = True,
        repetition_penalty: float = 1.15,
    ):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for internal attributes
        object.__setattr__(
            self,
            "tokenizer",
            AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            ),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Try to use flash_attention_2, fallback to default if not available
        model_kwargs = {
            "device_map": device_map,
            "dtype": torch_dtype,  # Use dtype instead of torch_dtype
            "trust_remote_code": True,
        }
        
        # Try flash_attention_2 first, fallback to default if it fails
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            object.__setattr__(
                self,
                "model",
                AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs),
            )
        except (ImportError, RuntimeError, ValueError) as e:
            # Flash Attention 2 not available or incompatible, use default
            print(f"Warning: Flash Attention 2 not available ({e}), using default attention implementation.", flush=True)
            model_kwargs.pop("attn_implementation", None)
            object.__setattr__(
                self,
                "model",
                AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs),
            )
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "concise_mode", concise_mode)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)

    # --- BaseChatModel required hooks ---
    @property
    def _llm_type(self) -> str:  # pragma: no cover - required by base class
        return "local-hf"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        stream = kwargs.pop("stream", False)
        if stream:
            # When called via .invoke(stream=True) we still want a final text output.
            collected: List[str] = []
            for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                try:
                    delta = (chunk.message.content or "")
                except Exception:
                    delta = ""
                if delta:
                    collected.append(delta)
            final_text = "".join(collected)
        else:
            final_text = self._run_once(messages, stop=stop, **kwargs)

        generation = ChatGeneration(message=AIMessage(content=final_text))
        return ChatResult(generations=[generation])

    # --- Internal helpers ---
    def _build_prompt(self, messages: List[BaseMessage]) -> str:
        chat_msgs = [{"role": _lc_to_hf_role(m), "content": getattr(m, "content", "")} for m in messages]
        
        # In concise mode, prepend a system instruction to be brief and avoid repetition
        if self.concise_mode:
            concise_instruction = {
                "role": "system",
                "content": (
                    "【输出风格要求】请用简洁精炼的语言回答，避免冗余和重复。"
                    "思考过程要简明扼要，直接给出关键结论和依据。"
                    "不要过度解释或重复已有信息。"
                )
            }
            # Insert at the beginning if no system message, or append to existing system
            if chat_msgs and chat_msgs[0]["role"] == "system":
                chat_msgs[0]["content"] = concise_instruction["content"] + "\n\n" + chat_msgs[0]["content"]
            else:
                chat_msgs.insert(0, concise_instruction)
        
        return self.tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_generation_args(self, **kwargs: Any) -> dict:
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        do_sample = True
        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = 1.0
            top_p = 1.0
        return {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
        }

    def _run_once(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_args = self._prepare_generation_args(**kwargs)
        output_ids = self.model.generate(
            **inputs,
            **generation_args,
        )[0][inputs["input_ids"].shape[-1] :]

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if stop:
            text = self._apply_stops(text, stop)
        return text

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> Iterable[ChatGenerationChunk]:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_args = self._prepare_generation_args(**kwargs)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            **generation_args,
        )

        worker = Thread(target=self.model.generate, kwargs=generation_kwargs)
        worker.start()

        collected: List[str] = []
        for token_text in streamer:
            collected.append(token_text)
            if run_manager is not None:
                try:
                    run_manager.on_llm_new_token(token_text)
                except Exception:
                    # Best-effort: callback support varies by LangChain version.
                    pass
            yield ChatGenerationChunk(message=AIMessageChunk(content=token_text))

        combined = "".join(collected)
        if stop:
            combined = self._apply_stops(combined, stop)

    @staticmethod
    def _apply_stops(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        for s in stop:
            if s and s in text:
                text = text.split(s, 1)[0]
        return text

    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,
    ) -> "LocalHFChatModelWithTools":
        """Bind tools to this model, returning a new model instance that can use them."""
        return LocalHFChatModelWithTools(
            base_model=self,
            tools=tools,
        )


class LocalVLLMChatModel(BaseChatModel):
    model_name: str = "local-vllm"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        model_path: str,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 0,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        concise_mode: bool = True,
        repetition_penalty: float = 1.15,
    ):
        super().__init__()
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise ImportError("未检测到 vllm，请先安装 vllm 以启用 FP8 本地加载。") from exc
        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len and max_model_len > 0:
            llm_kwargs["max_model_len"] = max_model_len
        object.__setattr__(self, "llm", LLM(**llm_kwargs))
        object.__setattr__(self, "tokenizer", self.llm.get_tokenizer())
        object.__setattr__(self, "_sampling_params_cls", SamplingParams)
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "concise_mode", concise_mode)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)

    @property
    def _llm_type(self) -> str:
        return "local-vllm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        stream = kwargs.pop("stream", False)
        if stream:
            collected: List[str] = []
            for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                try:
                    delta = (chunk.message.content or "")
                except Exception:
                    delta = ""
                if delta:
                    collected.append(delta)
            final_text = "".join(collected)
        else:
            final_text = self._run_once(messages, stop=stop, **kwargs)

        generation = ChatGeneration(message=AIMessage(content=final_text))
        return ChatResult(generations=[generation])

    def _build_prompt(self, messages: List[BaseMessage]) -> str:
        chat_msgs = [{"role": _lc_to_hf_role(m), "content": getattr(m, "content", "")} for m in messages]
        if self.concise_mode:
            concise_instruction = {
                "role": "system",
                "content": (
                    "【输出风格要求】请用简洁精炼的语言回答，避免冗余和重复。"
                    "思考过程要简明扼要，直接给出关键结论和依据。"
                    "不要过度解释或重复已有信息。"
                )
            }
            if chat_msgs and chat_msgs[0]["role"] == "system":
                chat_msgs[0]["content"] = concise_instruction["content"] + "\n\n" + chat_msgs[0]["content"]
            else:
                chat_msgs.insert(0, concise_instruction)
        return self.tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_sampling_params(self, stop: Optional[List[str]] = None, **kwargs: Any):
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if temperature is None or temperature <= 0:
            temperature = 0
            top_p = 1.0
        return self._sampling_params_cls(
            max_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            stop=stop or None,
        )

    def _run_once(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        prompt = self._build_prompt(messages)
        sampling_params = self._prepare_sampling_params(stop=stop, **kwargs)
        outputs = self.llm.generate([prompt], sampling_params)
        text = ""
        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text or ""
        if stop:
            text = self._apply_stops(text, stop)
        return text

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> Iterable[ChatGenerationChunk]:
        prompt = self._build_prompt(messages)
        sampling_params = self._prepare_sampling_params(stop=stop, **kwargs)
        try:
            generator = self.llm.generate([prompt], sampling_params, stream=True)
        except TypeError:
            text = self._run_once(messages, stop=stop, **kwargs)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            return

        prev_text = ""
        for output in generator:
            if not output.outputs:
                continue
            full_text = output.outputs[0].text or ""
            if full_text.startswith(prev_text):
                delta = full_text[len(prev_text):]
            else:
                delta = full_text
            prev_text = full_text
            if delta:
                if run_manager is not None:
                    try:
                        run_manager.on_llm_new_token(delta)
                    except Exception:
                        pass
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

    @staticmethod
    def _apply_stops(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        for s in stop:
            if s and s in text:
                text = text.split(s, 1)[0]
        return text

    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,
    ) -> "LocalHFChatModelWithTools":
        return LocalHFChatModelWithTools(
            base_model=self,
            tools=tools,
        )


class LocalHFChatModelWithTools(BaseChatModel):
    """Wrapper around LocalHFChatModel that supports tool calling."""

    model_name: str = "local-hf-with-tools"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        base_model: LocalHFChatModel,
        tools: List[BaseTool],
    ):
        super().__init__()
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_tools", tools)
        object.__setattr__(self, "_tools_by_name", {tool.name: tool for tool in tools if hasattr(tool, 'name')})

    @property
    def _llm_type(self) -> str:
        return "local-hf-with-tools"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        # Add tools information to the system message
        tools_prompt = self._build_tools_prompt()
        
        # Prepend tools info to messages if not already present
        enhanced_messages = list(messages)
        if tools_prompt and (not enhanced_messages or enhanced_messages[0].type != "system"):
            enhanced_messages.insert(0, lc_messages.SystemMessage(content=tools_prompt))
        elif tools_prompt and enhanced_messages[0].type == "system":
            # Append to existing system message
            existing_content = getattr(enhanced_messages[0], "content", "")
            enhanced_messages[0] = lc_messages.SystemMessage(
                content=f"{existing_content}\n\n{tools_prompt}"
            )

        # Generate response using base model
        stream = kwargs.pop("stream", False)
        if stream:
            collected: List[str] = []
            for chunk in self._base_model._stream(
                enhanced_messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                try:
                    delta = (chunk.message.content or "")
                except Exception:
                    delta = ""
                if delta:
                    collected.append(delta)
            final_text = "".join(collected)
        else:
            final_text = self._base_model._run_once(enhanced_messages, stop=stop, **kwargs)

        # Debug: print the full response text
        print(f"[DEBUG] Full model response text (length: {len(final_text)}): {final_text[:1000]}", flush=True)
        
        # Try to parse tool calls from the response
        tool_calls = self._parse_tool_calls(final_text)
        
        if tool_calls:
            # Create AIMessage with tool calls
            # Keep content if it's not just the tool call JSON
            content_to_keep = final_text
            # If the text is mostly just JSON, clear it
            if len(final_text.strip()) < 200 and ('{"tool_name"' in final_text or '{"name"' in final_text):
                content_to_keep = ""
            message = AIMessage(
                content=content_to_keep,
                tool_calls=tool_calls,
            )
            print(f"[DEBUG] Created AIMessage with {len(tool_calls)} tool calls", flush=True)
        else:
            message = AIMessage(content=final_text)
            print(f"[DEBUG] Created AIMessage without tool calls", flush=True)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> Iterable[ChatGenerationChunk]:
        """
        Streaming support so `prompt | model.bind_tools(...).stream(...)` works.
        Tool calls are parsed after streaming completes and emitted as a final chunk.
        """
        tools_prompt = self._build_tools_prompt()

        enhanced_messages = list(messages)
        if tools_prompt and (not enhanced_messages or enhanced_messages[0].type != "system"):
            enhanced_messages.insert(0, lc_messages.SystemMessage(content=tools_prompt))
        elif tools_prompt and enhanced_messages[0].type == "system":
            existing_content = getattr(enhanced_messages[0], "content", "")
            enhanced_messages[0] = lc_messages.SystemMessage(
                content=f"{existing_content}\n\n{tools_prompt}"
            )

        collected: List[str] = []
        for chunk in self._base_model._stream(
            enhanced_messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            try:
                delta = (chunk.message.content or "")
            except Exception:
                delta = ""
            if delta:
                collected.append(delta)
            yield chunk

        combined = "".join(collected)
        if stop:
            combined = self._apply_stops(combined, stop)

        tool_calls = self._parse_tool_calls(combined)
        if tool_calls:
            # Emit a final chunk carrying the parsed tool calls so downstream
            # nodes can execute tools even in streaming mode.
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=tool_calls))

    @staticmethod
    def _apply_stops(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        for s in stop:
            if s and s in text:
                text = text.split(s, 1)[0]
        return text

    def _build_tools_prompt(self) -> str:
        """Build a prompt describing available tools."""
        if not self._tools:
            return ""
        
        tools_desc = []
        
        # Add concise mode instruction if base model has it enabled
        if getattr(self._base_model, "concise_mode", False):
            tools_desc.append(
                "【输出风格】请简洁回答，避免冗余解释。需要调用工具时直接输出JSON，无需解释原因。"
            )
            tools_desc.append("")
        
        tools_desc.append("可用工具 (Available Tools):")
        tools_desc.append("")
        
        for tool in self._tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                tools_desc.append(f"- {tool.name}: {tool.description}")
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, "schema") else {}
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                if properties:
                    param_list = []
                    for k, v in properties.items():
                        is_required = k in required
                        param_type = v.get('type', 'str')
                        param_desc = f"{k}({param_type})"
                        if is_required:
                            param_desc += "【必需】"
                        param_list.append(param_desc)
                    params = ", ".join(param_list)
                    tools_desc.append(f"  参数: {params}")
        
        tools_desc.append("")
        tools_desc.append("=" * 50)
        tools_desc.append("工具调用格式 (Tool Call Format):")
        tools_desc.append("")
        tools_desc.append('必须严格按照以下 JSON 格式输出，不要添加任何其他文字：')
        tools_desc.append('')
        tools_desc.append('{"tool_name": "工具名称", "arguments": {"参数名": "参数值"}}')
        tools_desc.append('')
        tools_desc.append('示例 (Example):')
        tools_desc.append('{"tool_name": "search_drug_online", "arguments": {"drug_name": "维生素C", "info_type": "all"}}')
        tools_desc.append('')
        tools_desc.append("⚠️ 重要提示：")
        tools_desc.append("1. arguments 字段必须是完整的 JSON 对象，包含所有必需参数")
        tools_desc.append("2. arguments 不能为空 {}，必须提供所有必需参数的值")
        tools_desc.append("3. 如果需要调用工具，只输出 JSON，不要添加其他解释文字")
        tools_desc.append("=" * 50)
        
        return "\n".join(tools_desc)

    def _parse_tool_calls(self, text: str) -> List[dict]:
        """Parse tool calls from model output."""
        tool_calls = []
        
        # Debug: print raw text if it might contain tool calls
        if '"tool_name"' in text or '"name"' in text:
            print(f"[DEBUG] Raw model output (first 500 chars): {text[:500]}", flush=True)
        
        # Method 1: Try to find complete JSON objects with balanced braces
        # This is more reliable than simple regex
        def find_json_objects(text: str) -> List[dict]:
            """Find all valid JSON objects in text."""
            objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Find matching closing brace
                    brace_count = 0
                    start = i
                    j = i
                    while j < len(text):
                        if text[j] == '{':
                            brace_count += 1
                        elif text[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                try:
                                    obj_str = text[start:j+1]
                                    parsed = json.loads(obj_str)
                                    if isinstance(parsed, dict):
                                        objects.append(parsed)
                                except json.JSONDecodeError:
                                    pass
                                i = j
                                break
                        j += 1
                i += 1
            return objects
        
        # Find all JSON objects
        json_objects = find_json_objects(text)
        if json_objects:
            print(f"[DEBUG] Found {len(json_objects)} JSON objects in text", flush=True)
        
        for obj in json_objects:
            print(f"[DEBUG] Parsed JSON object: {obj}", flush=True)
            if isinstance(obj, list):
                for item in obj:
                    if self._is_valid_tool_call(item):
                        formatted = self._format_tool_call(item)
                        print(f"[DEBUG] Formatted tool call: {formatted}", flush=True)
                        tool_calls.append(formatted)
            elif self._is_valid_tool_call(obj):
                formatted = self._format_tool_call(obj)
                print(f"[DEBUG] Formatted tool call: {formatted}", flush=True)
                tool_calls.append(formatted)
        
        # Method 2: Also try to find function call patterns in code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
                print(f"[DEBUG] Found JSON in code block: {parsed}", flush=True)
                if isinstance(parsed, list):
                    for item in parsed:
                        if self._is_valid_tool_call(item):
                            formatted = self._format_tool_call(item)
                            print(f"[DEBUG] Formatted tool call from code block: {formatted}", flush=True)
                            tool_calls.append(formatted)
                elif self._is_valid_tool_call(parsed):
                    formatted = self._format_tool_call(parsed)
                    print(f"[DEBUG] Formatted tool call from code block: {formatted}", flush=True)
                    tool_calls.append(formatted)
            except json.JSONDecodeError:
                continue
        
        if tool_calls:
            print(f"[DEBUG] Final tool_calls: {tool_calls}", flush=True)
        else:
            print(f"[DEBUG] No tool calls found in text", flush=True)
        
        return tool_calls

    def _is_valid_tool_call(self, obj: dict) -> bool:
        """Check if an object is a valid tool call."""
        if not isinstance(obj, dict):
            return False
        tool_name = obj.get("tool_name") or obj.get("name")
        if not tool_name or tool_name not in self._tools_by_name:
            return False
        return True

    def _format_tool_call(self, obj: dict) -> dict:
        """Format a parsed tool call into LangChain's tool_calls format."""
        tool_name = obj.get("tool_name") or obj.get("name")
        arguments = obj.get("arguments") or obj.get("args") or {}
        
        print(f"[DEBUG] _format_tool_call - input obj: {obj}", flush=True)
        print(f"[DEBUG] _format_tool_call - extracted tool_name: {tool_name}, arguments: {arguments}", flush=True)
        
        # Ensure arguments is a dict
        if not isinstance(arguments, dict):
            print(f"[DEBUG] Arguments is not a dict, converting. Type: {type(arguments)}, Value: {arguments}", flush=True)
            arguments = {}
        
        # Validate required arguments if tool is available
        if tool_name and tool_name in self._tools_by_name:
            tool = self._tools_by_name[tool_name]
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, "schema") else {}
                required = schema.get("required", [])
                missing = [field for field in required if field not in arguments or not arguments[field]]
                if missing:
                    # If required arguments are missing, log a warning but still return the call
                    # The tool execution will handle the validation error
                    print(
                        f"[Warning] Tool call '{tool_name}' is missing required arguments: {missing}. "
                        f"Provided arguments: {arguments}. Original obj: {obj}",
                        flush=True
                    )
        
        result = {
            "name": tool_name,
            "args": arguments,
            "id": f"call_{tool_name}_{hash(str(arguments))}",
        }
        print(f"[DEBUG] _format_tool_call - result: {result}", flush=True)
        return result
