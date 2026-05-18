from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    Sequence,
    TypedDict,
    Unpack,
    overload,
    override,
)
from uuid import UUID

import httpx
from google.genai import Client
from google.genai.types import HttpOptions
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

try:
    # langchain-openai supports Azure OpenAI via AzureChatOpenAI
    from langchain_openai import AzureChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AzureChatOpenAI = None  # type: ignore
from pydantic import SecretStr
from tenacity import (
    after_log,
    stop_after_attempt,
    wait_random_exponential,
)

from ai2i.chain.computation import ChainComputation, ModelRunnable
from ai2i.chain.gemini.async_genai import AsyncChatGoogleGenAI
from ai2i.chain.models import LLMModel, ModelFamily
from ai2i.chain.params import DEFAULT_BATCH_MAX_CONCURRENCY
from ai2i.chain.retry import (
    MaboolTimeout,
    RacingRetrySettings,
    RacingRetryWithTenacity,
    RetryWithTenacity,
    TenacityRetrySettings,
)

default_logger = logging.getLogger(__name__)


class Timeouts:
    """Predefined timeout configurations for LLM requests.

    Use these based on expected model response time:
    - micro/tiny: Simple, fast completions
    - short: Most standard requests (default)
    - medium/long: Complex prompts or long outputs
    - reasoning: Reasoning models with extended thinking time
    """

    micro: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(2.0, read=2.0, write=2.0, connect=2.0))
    tiny: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(5.0, read=5.0, write=3.0, connect=3.0))
    short: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(10.0, read=10.0, write=5.0, connect=15.0))
    medium: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(30.0, read=30.0, write=15.0, connect=15.0))
    long: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(30.0, read=30.0, write=15.0, connect=15.0))
    extra_long: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(60.0, read=60.0, write=30.0, connect=30.0))
    reasoning: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(300.0, read=300.0, write=30.0, connect=30.0))


def default_retry_settings(logger: logging.Logger | None = None) -> TenacityRetrySettings:
    return TenacityRetrySettings(
        {
            # mabool defaults
            "wait": wait_random_exponential(multiplier=0.5, max=10),
            "stop": stop_after_attempt(5),
            "after": after_log(logger if logger is not None else default_logger, logging.DEBUG),
        }
    )


class BatchExecutionCallback:
    def on_llm_start(self) -> None:
        pass

    def on_llm_end(self) -> None:
        pass


class BatchExecutionContextBase(TypedDict, total=False):
    max_concurrency: int
    callbacks: Sequence[BatchExecutionCallback]


@dataclass
class _LangchainBatchCallbackAdapter(BaseCallbackHandler):
    chain_callback: BatchExecutionCallback

    @override
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.chain_callback.on_llm_start()

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.chain_callback.on_llm_end()


# We separate return_exceptions because this property influances the return type from a batch call
# so it will need to be treated differently from the other props in function signatures
class BatchExecutionContext(BatchExecutionContextBase, total=False):
    return_exceptions: bool


def _default_batch_execution_context() -> BatchExecutionContext:
    return {"max_concurrency": DEFAULT_BATCH_MAX_CONCURRENCY, "return_exceptions": False, "callbacks": []}


class ModelFactory(Protocol):
    def __call__(
        self,
        structured_response: bool,
        model: LLMModel[Any],
        timeout: MaboolTimeout,
        api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
    ) -> BaseChatModel: ...


def _create_chat_model(
    structured_response: bool,
    model: LLMModel[Any],
    timeout: MaboolTimeout,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
) -> BaseChatModel:
    """Create chat model instance, dispatching based on model family.

    The `openai` family auto-routes to **AzureChatOpenAI** when the
    `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_VERSION` /
    `AZURE_OPENAI_DEPLOYMENT` env vars are all present. This is how
    our Azure App Service deployment uses Azure OpenAI without
    changing any `openai:*` model names in the rest of the codebase
    - the dispatch happens here based purely on environment.
    Falls back to vanilla `ChatOpenAI` when any required Azure var
    is missing (local dev with the public OpenAI API still works).
    """
    if api_key_mapper is None:
        api_key_mapper = lambda _: None

    if model.family == "openai":
        params = model.to_api_params()
        if structured_response:
            params["model_kwargs"] = {"response_format": {"type": "json_object"}}

        # Azure OpenAI auto-detection. Check the standard Azure SDK env
        # var names first, then the older "OPENAI_*" variants some
        # tools use. All three must be set + `AzureChatOpenAI` must be
        # importable for us to take the Azure path.
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        resolved_api_key = api_key_mapper(model.family)

        if azure_endpoint and azure_api_version and azure_deployment and AzureChatOpenAI is not None:
            effective_key = SecretStr(azure_api_key) if azure_api_key else resolved_api_key

            # Some Azure OpenAI deployments (notably reasoning-model
            # deployments) only accept the default `temperature=1`. If
            # the caller set `temperature=0.0/0.1` for structured
            # extraction, Azure rejects with HTTP 400. Strip the
            # override so the deployment's default is used instead.
            azure_params = dict(params)
            if azure_params.get("temperature") is not None and azure_params.get("temperature") != 1:
                azure_params["temperature"] = None

            return AzureChatOpenAI(  # type: ignore[call-arg]
                api_key=effective_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                azure_deployment=azure_deployment,
                timeout=timeout.httpx_timeout,
                **azure_params,
            )

        return ChatOpenAI(
            api_key=resolved_api_key,
            model=model.name,
            timeout=timeout.httpx_timeout,
            use_responses_api=True,
            **params,
        )
    else:  # model.family == "google"
        params = model.to_api_params()
        if structured_response:
            params["generation_config"] = {
                **params.get("generation_config", {}),
                "response_mime_type": "application/json",
            }
        timeout_seconds = timeout.total_seconds()
        api_key = api_key_mapper(model.family)
        secret_api_key = str(api_key.get_secret_value()) if api_key else None
        return AsyncChatGoogleGenAI(
            model_name=model.name,
            client=Client(
                api_key=secret_api_key,
                http_options=HttpOptions(timeout=int(1000 * timeout_seconds) if timeout_seconds is not None else None),
            ),
            model_kwargs=params,
        )


type TrueLiteral = Literal[True]
type FalseLiteral = Literal[False]


@dataclass(frozen=True)
class Execution[IN, OUT]:
    runnable: Runnable[IN, OUT]

    async def once(self, input: IN) -> OUT:
        return await self.runnable.ainvoke(input)

    # The following overload changes the return type based on the 'return_exception' parameter. If it's set to 'True'
    # The resulting list will contain elements of 'OUT | Exception', if it's False it will only contain 'OUT' elements
    @overload
    async def many(
        self, inputs: list[IN], *, return_exceptions: TrueLiteral, **batch_ec: Unpack[BatchExecutionContextBase]
    ) -> list[OUT | Exception]:
        pass

    @overload
    async def many(
        self,
        inputs: list[IN],
        *,
        return_exceptions: FalseLiteral = False,
        **batch_ec: Unpack[BatchExecutionContextBase],
    ) -> list[OUT]:
        pass

    async def many(
        self, inputs: list[IN], return_exceptions: bool = False, **batch_ec: Unpack[BatchExecutionContextBase]
    ) -> list[OUT | Exception] | list[OUT]:
        resolved_batch_ec: BatchExecutionContext = {
            **_default_batch_execution_context(),
            **batch_ec,
            "return_exceptions": return_exceptions,
        }

        callbacks: list[BaseCallbackHandler] = [
            _LangchainBatchCallbackAdapter(c) for c in resolved_batch_ec.get("callbacks", [])
        ]

        return await self.runnable.abatch(
            inputs,
            config={
                "max_concurrency": resolved_batch_ec.get("max_concurrency"),
                "callbacks": callbacks,
            },
            return_exceptions=resolved_batch_ec["return_exceptions"],
        )


@dataclass(frozen=True)
class LLMEndpoint:
    """Configuration for executing LLM chains with retry and timeout settings.

    The appropriate chat model is created at execution time based on the model's family.
    """

    default_retry_settings: TenacityRetrySettings
    default_timeout: MaboolTimeout
    default_model: LLMModel[Any]
    model_factory: ModelFactory = _create_chat_model
    api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None
    default_racing_retry_settings: RacingRetrySettings | None = None

    def timeout(self, timeout: MaboolTimeout) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings,
            timeout,
            self.default_model,
            self.model_factory,
            self.api_key_mapper,
            self.default_racing_retry_settings,
        )

    def model(self, model: LLMModel[Any]) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings,
            self.default_timeout,
            model,
            self.model_factory,
            self.api_key_mapper,
            self.default_racing_retry_settings,
        )

    def retry_settings(self, **kwargs: Any) -> LLMEndpoint:
        return LLMEndpoint(
            TenacityRetrySettings({**self.default_retry_settings, **kwargs}),
            self.default_timeout,
            self.default_model,
            self.model_factory,
            self.api_key_mapper,
            self.default_racing_retry_settings,
        )

    def with_api_key_mapper(self, api_key_mapper: Callable[[ModelFamily], SecretStr | None]) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings,
            self.default_timeout,
            self.default_model,
            self.model_factory,
            api_key_mapper,
            self.default_racing_retry_settings,
        )

    def racing_retry_settings(
        self,
        soft_timeout_ratio: float = 0.33,
        fallback_models: list[LLMModel[Any]] | None = None,
        cancel_pending: bool = False,
    ) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings,
            self.default_timeout,
            self.default_model,
            self.model_factory,
            self.api_key_mapper,
            RacingRetrySettings(
                soft_timeout_ratio=soft_timeout_ratio,
                fallback_models=fallback_models if fallback_models is not None else [],
                cancel_pending=cancel_pending,
            ),
        )

    def execute[IN, OUT](self, computation: ChainComputation[IN, OUT]) -> Execution[IN, OUT]:
        def runnable_factory(is_structured: bool) -> ModelRunnable:
            chat_model = self.model_factory(
                is_structured, self.default_model, self.default_timeout, self.api_key_mapper
            )

            if self.default_racing_retry_settings is not None:
                fallback_runnables = []
                if self.default_racing_retry_settings.fallback_models:
                    for fallback_model in self.default_racing_retry_settings.fallback_models:
                        fallback_runnable = self.model_factory(
                            is_structured, fallback_model, self.default_timeout, self.api_key_mapper
                        )
                        fallback_runnables.append(fallback_runnable)

                return RacingRetryWithTenacity(
                    chat_model,
                    self.default_timeout.total_seconds() or 10.0,
                    self.default_retry_settings,
                    self.default_racing_retry_settings,
                    fallback_runnables,
                )
            else:
                return RetryWithTenacity(chat_model, self.default_retry_settings)

        runnable = computation.build_runnable(runnable_factory)
        return Execution(runnable)


def define_llm_endpoint(
    *,
    default_model: LLMModel[Any] | None = None,
    default_timeout: MaboolTimeout | None = None,
    logger: logging.Logger | None = None,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
    racing_soft_timeout_ratio: float | None = None,
    racing_fallback_model: list[LLMModel[Any]] | None = None,
    racing_cancel_pending: bool = True,
    **retry_settings: Any,
) -> LLMEndpoint:
    racing_settings = None
    if racing_soft_timeout_ratio is not None:
        racing_settings = RacingRetrySettings(
            soft_timeout_ratio=racing_soft_timeout_ratio,
            fallback_models=racing_fallback_model or [],
            cancel_pending=racing_cancel_pending,
        )

    if default_model is None:
        default_model = LLMModel.gpt4o()

    if default_timeout is None:
        if default_model.is_reasoning:
            default_timeout = Timeouts.reasoning
        else:
            default_timeout = Timeouts.short
    elif default_model.is_reasoning:
        reasoning_seconds = Timeouts.reasoning.httpx_timeout.read
        current_seconds = default_timeout.httpx_timeout.read
        if current_seconds is None or reasoning_seconds is None or current_seconds < reasoning_seconds:
            default_timeout = Timeouts.reasoning

    return LLMEndpoint(
        TenacityRetrySettings({**default_retry_settings(logger), **retry_settings}),
        default_timeout,
        default_model,
        _create_chat_model,
        api_key_mapper,
        racing_settings,
    )
