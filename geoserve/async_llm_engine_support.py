import logging

from typing import Optional, AsyncGenerator, Mapping

from ray import serve
from ray.serve.handle import DeploymentHandle

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams, BeamSearchParams
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from geoserve.async_task_runner import AsyncTaskRunner


@serve.deployment(name="AsyncLLMEngine", ray_actor_options={"num_gpus": 1, "num_cpus": 4})
class AsyncLLMEngineDeployment:
    """
    Ray serve deployment based on VLLM AsyncLLMEngine. We create this
    SO that we can easily scale this and wire it with other serving components
    """
    def __init__(
            self,
            engine_args: AsyncEngineArgs,
    ):
        # create logger
        self.logger = logging.getLogger("async_llm_support")
        # create engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.debug("Created AsyncLLMEngine")

    async def abort(self, request_id: str) -> None:
        """
        Abort request
        :param request_id: request id
        :return: None
        """
        self.logger.debug("AsyncLLMEngine - abort request")
        return await self.engine.abort(request_id=request_id)

    async def check_health(self) -> None:
        """
        Raises an error if engine is unhealthy.
        :return: None
        """
        self.logger.info("AsyncLLMEngine - check health request")
        await self.engine.check_health()

    async def generate(
            self,
            prompt: PromptType,
            sampling_params: SamplingParams,
            request_id: str,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.
        :param prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType
            `for more details about the format of each input.
        :param sampling_params: the sampling parameters of the request.
        :param request_id: the unique id of the request
        :param lora_request: LoRA request to use for generation, if any
        :param trace_headers: OpenTelemetry trace headers
        :param prompt_adapter_request: Prompt Adapter request to use for generation, if any
        :param priority: the priority of the request. Only applicable with priority scheduling.
        :return:
        """
        self.logger.info("AsyncLLMEngine - generate request")
        return self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id,
                                    lora_request=lora_request, trace_headers=trace_headers,
                                    prompt_adapter_request=prompt_adapter_request, priority=priority)

    async def beam_search(
            self,
            prompt: PromptType,
            request_id: str,
            params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        The beam_search method implements beam search on top of generate. For example, to search using 5 beams and
        output at most 50 tokens
        Unlike greedy search, beam-search decoding keeps several hypotheses at each time step and eventually chooses
        the hypothesis that has the overall highest probability for the entire sequence. This has the advantage of
        identifying high-probability sequences that start with lower probability initial tokens and would’ve been
        ignored by the greedy search.
        :param prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType
            `for more details about the format of each input.
        :param request_id: request id
        :param params: The beam search parameters
        :return:
        """
        self.logger.info("AsyncLLMEngine - beam search request")
        return self.engine.beam_search(prompt=prompt, request_id=request_id, params=params)

    async def encode(
            self,
            prompt: PromptType,
            pooling_params: PoolingParams,
            request_id: str,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            priority: int = 0,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """
        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.
        :param prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
            for more details about the format of each input.
        :param pooling_params: the pooling parameters of the request.
        :param request_id: the unique id of the request
        :param lora_request: LoRA request to use for generation, if any.
        :param trace_headers: OpenTelemetry trace headers
        :param priority: the priority of the request. Only applicable with priority scheduling.
        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.
        """
        self.logger.info("AsyncLLMEngine - encode request")
        return self.engine.encode(prompt=prompt, pooling_params=pooling_params, request_id=request_id,
                                  lora_request=lora_request, trace_headers=trace_headers, priority=priority)

    async def engine_step(self, virtual_engine: int) -> bool:
        """
        Kick the engine to process the waiting requests.
        :param virtual_engine: Virtual engine number
        :return: True if there are in-progress requests.
        """
        self.logger.info("AsyncLLMEngine - engine step request")
        return await self.engine.engine_step(virtual_engine=virtual_engine)

    async def get_decoding_config(self) -> DecodingConfig:
        """
        Get the decoding configuration of the vLLM engine
        :return: decoding config
        """
        self.logger.info("AsyncLLMEngine - get decoding config request")
        return await self.engine.get_decoding_config()

    async def get_lora_config(self) -> LoRAConfig:
        """
        Get the lora configuration of the vLLM engine
        :return: LoRA config
        """
        self.logger.info("AsyncLLMEngine - get lora config request")
        return await self.engine.get_lora_config()

    async def get_model_config(self) -> ModelConfig:
        """
        Get the model configuration of the vLLM engine.
        :return: Model config
        """
        self.logger.info("AsyncLLMEngine - get model config request")
        return await self.engine.get_model_config()

    async def get_parallel_config(self) -> ParallelConfig:
        """
        Get the parallel configuration of the vLLM engine.
        :return: parallel config
        """
        self.logger.info("AsyncLLMEngine - get parallel config request")
        return await self.engine.get_parallel_config()

    async def get_scheduler_config(self) -> SchedulerConfig:
        """
        Get the scheduling configuration of the vLLM engine.
        :return: scheduling config
        """
        self.logger.info("AsyncLLMEngine - get scheduler config request")
        return await self.engine.get_scheduler_config()

    async def get_tokenizer(
            self,
            lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """
        Get the appropriate tokenizer for the request
        :param lora_request: optional LoRA request
        :return: tokenizer
        """
        self.logger.info("AsyncLLMEngine - get tokenizer request")
        return await self.engine.get_tokenizer(lora_request=lora_request)

    def errored(self) -> bool:
        """
        Check if the engine in error
        :return:
        """
        self.logger.info("AsyncLLMEngine - get error status")
        return self.engine.errored

    async def is_tracing_enabled(self) -> bool:
        """
        Check if tracing is enable
        :return:
        """
        self.logger.info("AsyncLLMEngine - get tracing enabled request")
        return await self.engine.is_tracing_enabled()


class AsyncLLMEngineProxy:
    """
    This class is a proxy around AsyncLLMEngineDeployment that hides the fact that
    a deployment is a Ray Actor requiring remoting. We need this wrapper to be able to use
    OpenAIServingChat as is with the deployment
    """
    def __init__(self, engine: DeploymentHandle):
        self.runner = AsyncTaskRunner.get_instance()
        self.engine = engine

    async def is_tracing_enabled(self) -> bool:
        return await self.engine.is_tracing_enabled.remote()

    @property
    def errored(self) -> bool:
        return self.runner.get_serve_resp(self.engine.errored.remote())

    def get_model_config(self) -> ModelConfig:
        return self.runner.get_serve_resp(self.engine.get_model_config.remote())

    async def get_tokenizer(
            self,
            lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return await self.engine.get_tokenizer.remote(lora_request=lora_request)

    def generate(
            self,
            prompt: PromptType,
            sampling_params: SamplingParams,
            request_id: str,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        return self.engine.options(stream=True).generate.remote(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority
        )

    def beam_search(
            self,
            prompt: PromptType,
            request_id: str,
            params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        return self.engine.options(stream=True).beam_search.remote(
            prompt=prompt,
            request_id=request_id,
            params=params,
        )
