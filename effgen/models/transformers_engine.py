"""
HuggingFace Transformers engine implementation as fallback.

This module provides a fallback inference engine using HuggingFace Transformers
with features including:
- Automatic quantization with bitsandbytes
- Flash Attention support
- Multi-GPU device mapping
- Memory optimization techniques
- CPU fallback support
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import (
    GenerationConfig as HFGenerationConfig,
)

from effgen.models.base import (
    BatchModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='accelerate')
warnings.filterwarnings('ignore', message='.*Some parameters are on the meta device.*')

logger = logging.getLogger(__name__)


class TransformersEngine(BatchModel):
    """
    HuggingFace Transformers-based model engine.

    This engine serves as a fallback when vLLM is unavailable or incompatible.
    It supports a wider range of models and edge cases.

    Features:
    - Automatic quantization (4-bit, 8-bit with bitsandbytes)
    - Flash Attention 2 support
    - Auto device mapping for multi-GPU
    - Memory optimization (gradient checkpointing, mixed precision)
    - CPU fallback
    - Streaming generation

    Attributes:
        model_name: HuggingFace model identifier or path
        quantization_bits: Quantization level (None, 4, 8)
        device_map: Device mapping strategy ('auto', 'balanced', or custom)
        use_flash_attention: Whether to use Flash Attention 2
        torch_dtype: Torch data type for model weights
    """

    def __init__(
        self,
        model_name: str,
        quantization_bits: int | None = None,
        device_map: str = "auto",
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
        max_memory: dict[int, str] | None = None,
        offload_folder: str | None = None,
        **kwargs
    ):
        """
        Initialize Transformers engine.

        Args:
            model_name: HuggingFace model ID or local path
            quantization_bits: Quantization level (None, 4, or 8)
            device_map: Device mapping ('auto', 'balanced', 'sequential', or dict)
            use_flash_attention: Enable Flash Attention 2 if available
            torch_dtype: Data type (None for auto, or torch.float16, torch.bfloat16)
            trust_remote_code: Whether to trust remote code
            low_cpu_mem_usage: Use low CPU memory during loading
            max_memory: Maximum memory per device (e.g., {0: "20GB", "cpu": "30GB"})
            offload_folder: Folder for offloading weights
            **kwargs: Additional model loading arguments
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS
        )

        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_memory = max_memory
        self.offload_folder = offload_folder

        # Filter out parameters that shouldn't be passed to model loading
        # These are vLLM-specific or other incompatible parameters
        self.additional_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ['quantization', 'engine', 'backend', 'device',
                                               'use_tqdm', 'tensor_parallel_size',
                                               'apply_chat_template', 'system_prompt',
                                               'gpu_memory_utilization', 'max_num_seqs',
                                               'max_num_batched_tokens']}

        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> None:
        """
        Load the model using HuggingFace Transformers.

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        try:
            logger.info(f"Loading model '{self.model_name}' with Transformers...")

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
            else:
                self.device = "cpu"
                # If the host actually has NVIDIA GPUs but torch can't use them
                # (almost always a torch-CUDA vs driver mismatch), emit one clear,
                # actionable warning instead of a bland "CUDA not available".
                from effgen.gpu.cuda_compat import warn_cuda_mismatch_once
                if not warn_cuda_mismatch_once():
                    logger.warning("CUDA not available, using CPU (this will be slow)")

            # Setup quantization config if specified
            quantization_config = None
            if self.quantization_bits is not None:
                quantization_config = self._create_quantization_config()

            # Determine torch dtype
            if self.torch_dtype is None:
                if self.device == "cuda":
                    # Use bfloat16 if available, else float16
                    if torch.cuda.is_bf16_supported():
                        self.torch_dtype = torch.bfloat16
                    else:
                        self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32

            logger.info(
                f"Configuration: quantization={self.quantization_bits}-bit, "
                f"dtype={self.torch_dtype}, flash_attention={self.use_flash_attention}"
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Build model loading arguments
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }

            # Add quantization config if using quantization
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                # Don't set dtype when quantizing
            else:
                # transformers v5+ uses 'dtype', v4.x uses 'torch_dtype'
                import transformers
                if hasattr(transformers, 'VERSION') or int(transformers.__version__.split('.')[0]) >= 5:
                    model_kwargs["dtype"] = self.torch_dtype
                else:
                    model_kwargs["torch_dtype"] = self.torch_dtype

            # Add device map for multi-GPU or CPU offloading
            if self.device == "cuda":
                model_kwargs["device_map"] = self.device_map

                if self.max_memory:
                    model_kwargs["max_memory"] = self.max_memory

                if self.offload_folder:
                    model_kwargs["offload_folder"] = self.offload_folder

            # Add Flash Attention 2 if requested
            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Add additional kwargs
            model_kwargs.update(self.additional_kwargs)

            # Load model
            try:
                # Suppress Flash Attention warnings from transformers
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                    warnings.filterwarnings('ignore', message='.*flash_attn.*')
                    self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            except Exception as e:
                # Fallback without Flash Attention if it fails
                if self.use_flash_attention and "flash" in str(e).lower():
                    logger.info("Flash Attention not available, using standard attention")
                    model_kwargs.pop("attn_implementation", None)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                        warnings.filterwarnings('ignore', message='.*flash_attn.*')
                        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                else:
                    raise

            # Move to device if not using device_map
            if "device_map" not in model_kwargs and self.device != "cpu":
                self.model = self.model.to(self.device)

            # Set model to eval mode
            self.model.eval()

            # Store metadata
            self._context_length = self._get_max_length()
            self._metadata = {
                "model_name": self.model_name,
                "quantization": f"{self.quantization_bits}-bit" if self.quantization_bits else None,
                "dtype": str(self.torch_dtype),
                "device": str(self.device),
                "flash_attention": self.use_flash_attention,
                "max_length": self._context_length,
                "num_parameters": self.model.num_parameters(),
            }

            self._is_loaded = True
            logger.info(f"Model '{self.model_name}' loaded successfully with Transformers")

        except Exception as e:
            logger.error(f"Failed to load model with Transformers: {e}")
            raise RuntimeError(f"Transformers model loading failed: {e}") from e

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration.

        Returns:
            BitsAndBytesConfig for bitsandbytes quantization

        Raises:
            ValueError: If quantization_bits is invalid
        """
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(
                f"Invalid quantization_bits: {self.quantization_bits}. "
                "Must be 4 or 8."
            )

    def _get_max_length(self) -> int:
        """
        Get maximum context length from model config.

        Returns:
            int: Maximum sequence length
        """
        config = self.model.config

        # Try different config attributes
        for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
            if hasattr(config, attr):
                return getattr(config, attr)

        # Some models (e.g. gemma-3) nest config inside text_config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
                if hasattr(text_config, attr):
                    return getattr(text_config, attr)

        logger.warning("Could not determine max length from config, using 2048")
        return 2048

    def _create_generation_config(
        self,
        config: GenerationConfig | None = None
    ) -> tuple[HFGenerationConfig, list[str]]:
        """
        Create HuggingFace GenerationConfig from our GenerationConfig.

        Args:
            config: Our generation configuration

        Returns:
            Tuple of (HuggingFace GenerationConfig object, stop_sequences list)

        Notes:
            HuggingFace Transformers doesn't support stop sequences natively like OpenAI,
            so we return them separately for post-generation processing.
        """
        if config is None:
            config = GenerationConfig()

        hf_config = HFGenerationConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_new_tokens=config.max_tokens or 512,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Return stop sequences separately for post-processing
        # NOTE: We DON'T set them as eos_token_id because that would stop generation
        # at the first token match, not the full sequence match
        return hf_config, config.stop_sequences if config.stop_sequences else []

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        try:
            # Extract tool definitions before sanitizing kwargs — these are
            # passed to the chat template, not to HF generate()
            tools_for_template = kwargs.pop("tools", None)

            # Sanitize kwargs for HuggingFace Transformers compatibility
            # Convert OpenAI-style parameters to HuggingFace format
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                try:
                    if key == "max_tokens":
                        # Convert max_tokens to max_new_tokens for Transformers
                        sanitized_kwargs["max_new_tokens"] = value
                        logger.debug(f"Converted max_tokens={value} to max_new_tokens={value}")
                    elif key in ["temperature", "top_p", "top_k", "repetition_penalty",
                                "num_beams", "do_sample", "pad_token_id", "eos_token_id"]:
                        # These are valid HuggingFace parameters
                        sanitized_kwargs[key] = value
                    else:
                        # Log and skip unknown parameters to avoid errors
                        logger.warning(f"Skipping unknown generation parameter: {key}={value}")
                except Exception as e:
                    logger.error(f"Error processing generation parameter {key}: {e}")
                    # Continue processing other parameters
                    continue

            # Apply chat template if available for better model compatibility
            # Many modern models like Qwen expect a specific format
            formatted_prompt = prompt
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                try:
                    # Wrap prompt as a user message for chat models
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    template_kwargs: dict[str, Any] = {
                        "tokenize": False,
                        "add_generation_prompt": True,
                    }
                    # Pass tool definitions for native function calling
                    if tools_for_template:
                        template_kwargs["tools"] = tools_for_template
                        logger.debug(
                            f"Passing {len(tools_for_template)} tool definitions "
                            "to chat template for native function calling"
                        )
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        **template_kwargs,
                    )
                    logger.debug("Applied chat template to prompt")
                except TypeError as e:
                    # Template may not support tools param — fall back without tools
                    if tools_for_template:
                        logger.debug(
                            f"Chat template does not accept tools param: {e}, "
                            "falling back to plain template"
                        )
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        logger.warning(f"Failed to apply chat template: {e}")
                        formatted_prompt = prompt
                except Exception as e:
                    logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
                    formatted_prompt = prompt

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with sanitized kwargs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **sanitized_kwargs
                )

            # Decode output
            # When native tool calling is active, preserve tool-call tokens
            # like <tool_call>, </tool_call>, [TOOL_CALLS] etc. but strip
            # chat-template end markers like <|im_end|>, </s>, <|eot_id|>.
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            if tools_for_template:
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False,
                )
                # Strip common chat-template end tokens
                for end_marker in ("<|im_end|>", "</s>", "<|eot_id|>",
                                   "<|end_of_text|>", "<|endoftext|>"):
                    generated_text = generated_text.replace(end_marker, "")
                generated_text = generated_text.strip()
            else:
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                )

            # Apply stop sequences post-generation
            # This is more reliable than trying to use them during generation
            finish_reason = "stop"
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        # Find first occurrence of any stop sequence
                        stop_index = generated_text.find(stop_seq)
                        if stop_index != -1:
                            generated_text = generated_text[:stop_index]
                            finish_reason = "stop_sequence"
                            logger.debug(f"Stopped generation at stop sequence: '{stop_seq}'")
                            break

            # Calculate tokens
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(generated_ids)

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "stop_sequences_applied": stop_sequences if stop_sequences else []
                }
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use TextIteratorStreamer for streaming
            from threading import Event, Thread

            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=30.0,  # prevent indefinite block if generation thread dies
            )

            stop_event = Event()

            class _StopOnEvent(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    return stop_event.is_set()

            stopping_criteria = kwargs.pop("stopping_criteria", None)
            if stopping_criteria is None:
                stopping_criteria = StoppingCriteriaList([_StopOnEvent()])
            elif isinstance(stopping_criteria, StoppingCriteriaList):
                stopping_criteria.append(_StopOnEvent())
            else:
                try:
                    criteria_items = list(stopping_criteria)
                except TypeError:
                    criteria_items = [stopping_criteria]
                stopping_criteria = StoppingCriteriaList([*criteria_items, _StopOnEvent()])

            # Note: stop_sequences not fully supported in streaming mode
            # They would need to be checked in the consumer of the stream

            # Generate in a separate thread
            generation_kwargs = {
                **inputs,
                "generation_config": generation_config,
                "stopping_criteria": stopping_criteria,
                "streamer": streamer,
                **kwargs
            }

            gen_exception: list[BaseException] = []

            def _generate_with_error_capture() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except BaseException as exc:
                    gen_exception.append(exc)
                    # Unblock the streamer queue so yield-from terminates
                    streamer.end()

            thread = Thread(target=_generate_with_error_capture, daemon=False)
            thread.start()

            try:
                # Yield tokens as they're generated
                yield from streamer
            finally:
                stop_event.set()
                try:
                    streamer.end()
                except Exception:
                    logger.debug("Failed to end streamer", exc_info=True)
                thread.join(timeout=30.0)

            if thread.is_alive():
                raise RuntimeError("Streaming generation thread did not exit cleanly")
            if gen_exception:
                raise gen_exception[0]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Validate all prompts
        for prompt in prompts:
            self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        try:
            # Tokenize all inputs
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **kwargs
                )

            # Decode outputs
            results = []
            for i, output in enumerate(outputs):
                # Get only the generated part (exclude input)
                prompt_length = inputs["input_ids"][i].shape[0]
                generated_ids = output[prompt_length:]

                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                results.append(GenerationResult(
                    text=generated_text,
                    tokens_used=len(generated_ids),
                    finish_reason="stop",
                    model_name=self.model_name,
                    metadata={
                        "prompt_tokens": prompt_length,
                        "completion_tokens": len(generated_ids),
                        "total_tokens": prompt_length + len(generated_ids),
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise RuntimeError(f"Batch generation failed: {e}") from e

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return TokenCount(count=len(tokens), model_name=self.model_name)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise RuntimeError(f"Token counting failed: {e}") from e

    def get_context_length(self) -> int:
        """
        Get maximum context length.

        Returns:
            int: Maximum context length in tokens
        """
        if self._context_length is not None:
            return self._context_length
        return 2048  # Default fallback

    def get_max_batch_size(self) -> int:
        """
        Get maximum batch size.

        Returns:
            int: Maximum batch size (conservative estimate)
        """
        # Conservative batch size based on available VRAM
        if self.device == "cuda":
            return 8
        else:
            return 1  # CPU is slow, use minimal batch size

    def supports_tool_calling(self) -> bool:
        """Check if the model supports native tool calling via chat template.

        Returns True if the tokenizer's chat template accepts a ``tools``
        parameter, which is the case for Qwen2.5, Llama 3.x, Mistral, etc.
        """
        if not self._is_loaded or self.tokenizer is None:
            return False
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            return False
        # Test whether the chat template accepts a `tools` kwarg
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "test",
                        "description": "test",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except (TypeError, Exception):
            return False

    def unload(self) -> None:
        """
        Unload the model and free memory.

        Removes any accelerate device-dispatch hooks before deleting the
        model — leftover hooks can corrupt the CUDA forward state of
        subsequently-loaded models in the same process (observed as
        intermittent C-level aborts inside Qwen2 RMSNorm under pytest).
        """
        if self.model is not None:
            logger.info(f"Unloading model '{self.model_name}'...")
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.model, recurse=True)
            except Exception:
                logger.debug("Failed to remove accelerate hooks during unload", exc_info=True)
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug("torch.cuda.synchronize() failed during unload", exc_info=True)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                logger.debug("torch.cuda.ipc_collect() failed during unload", exc_info=True)

        self._is_loaded = False
        logger.info(f"Model '{self.model_name}' unloaded successfully")
