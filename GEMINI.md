# Gemini Models Documentation

This document provides an overview of the `GeminiModel` and `Gemini` classes, which facilitate interaction with Gemini large language models.

## `GeminiModel` Class

The `GeminiModel` class provides a simplified interface for calling Gemini models, including retry logic and parallel request execution.

### Initialization (`__init__`)

```python
class GeminiModel:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        finetuned_model: bool = False,
        distribute_requests: bool = False,
        cache_name: str | None = None,
        temperature: float = 0.01,
        **kwargs,
    ):
        # ... (implementation details)
```

**Parameters:**

- `model_name` (`str`, optional): The name of the Gemini model to use (default: `"gemini-2.0-flash-001"`).
- `finetuned_model` (`bool`, optional): If `True`, indicates that a finetuned model is being used (default: `False`).
- `distribute_requests` (`bool`, optional): If `True`, distributes requests across available Gemini regions (default: `False`).
- `cache_name` (`str | None`, optional): A name for caching content. If provided, the model will be loaded from cached content (default: `None`).
- `temperature` (`float`, optional): The temperature for generation, controlling randomness (default: `0.01`).
- `**kwargs`: Additional arguments passed to the `GenerationConfig`.

### `call` Method

```python
    @retry(max_attempts=12, base_delay=2, backoff_factor=2)
    def call(self, prompt: str, parser_func=None) -> str:
        """Calls the Gemini model with the given prompt.

        Args:
            prompt (str): The prompt to call the model with.
            parser_func (callable, optional): A function that processes the LLM
              output. It takes the model"s response as input and returns the
              processed result.

        Returns:
            str: The processed response from the model.
        """
        # ... (implementation details)
```

This method calls the Gemini model with a single prompt, applying retry logic. It can optionally process the model's response using a `parser_func`.

### `call_parallel` Method

```python
    def call_parallel(
        self,
        prompts: List[str],
        parser_func: Optional[Callable[[str], str]] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ) -> List[Optional[str]]:
        """Calls the Gemini model for multiple prompts in parallel using threads with retry logic.

        Args:
            prompts (List[str]): A list of prompts to call the model with.
            parser_func (callable, optional): A function to process each response.
            timeout (int): The maximum time (in seconds) to wait for each thread.
            max_retries (int): The maximum number of retries for timed-out threads.

        Returns:
            List[Optional[str]]:
            A list of responses, or None for threads that failed.
        """
        # ... (implementation details)
```

This method enables parallel execution of multiple prompts to the Gemini model, utilizing threads and incorporating retry logic for robustness.

## `Gemini` Class

The `Gemini` class provides an integration for Gemini models, extending `BaseLlm` for advanced functionalities like asynchronous content generation and API client management.

### Supported Models (`supported_models`)

```python
  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """
    return [
        r'gemini-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]
```

This static method returns a list of regular expressions defining the patterns for supported Gemini models, including generic Gemini models, fine-tuned Vertex AI endpoints, and Vertex AI Gemini long names.

### `generate_content_async` Method

```python
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    # ... (implementation details)
```

This asynchronous method sends a request to the Gemini model. It supports both streaming and non-streaming responses, handling preprocessing, tracking headers, and response parsing.

### `gemini_llm` Function

```python
def gemini_llm():
  return Gemini(model="gemini-1.5-flash")
```

This helper function returns an instance of the `Gemini` class, initialized with the `"gemini-1.5-flash"` model.

## Internal Helpers

### `_preprocess_request` Method

```python
def _preprocess_request(self, llm_request: LlmRequest) -> None:
    if self._api_backend == GoogleLLMVariant.GEMINI_API:
      # Using API key from Google AI Studio to call model doesn't support labels.
      if llm_request.config:
        llm_request.config.labels = None

      if llm_request.contents:
        for content in llm_request.contents:
          if not content.parts:
            continue
          for part in content.parts:
            _remove_display_name_if_present(part.inline_data)
            _remove_display_name_if_present(part.file_data)
```

This internal method preprocesses the `llm_request` before sending it to the Gemini model. It specifically handles cases where the API key from Google AI Studio is used, which does not support labels, and removes display names from inline or file data parts.

### `_api_backend` Property

```python
def _api_backend(self) -> GoogleLLMVariant:
    return (
        GoogleLLMVariant.VERTEX_AI
        if self.api_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )
```

This cached property determines the API backend (Vertex AI or Gemini API) based on the `api_client` configuration.

### `_to_gemini_schema` Function

```python
def _to_gemini_schema(openapi_schema: dict[str, Any]) -> Schema:
  """Converts an OpenAPI schema dictionary to a Gemini Schema object."""
  if openapi_schema is None:
    return None

  if not isinstance(openapi_schema, dict):
    raise TypeError("openapi_schema must be a dictionary")

  openapi_schema = _sanitize_schema_formats_for_gemini(openapi_schema)
  return Schema.from_json_schema(
      json_schema=_ExtendedJSONSchema.model_validate(openapi_schema),
      api_option=get_google_llm_variant(),
  )
```

This function converts an OpenAPI schema dictionary into a Gemini `Schema` object, sanitizing it for Gemini compatibility.

### `gemini_to_json_schema` Function

```python
def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary.

  Args:
      gemini_schema: An instance of the Gemini Schema class.

  Returns:
      A dictionary representing the equivalent JSON Schema.

  Raises:
      TypeError: If the input is not an instance of the expected Schema class.
      ValueError: If an invalid Gemini Type enum value is encountered.
  """
  # ... (implementation details)
```

This function converts a Gemini `Schema` object into its equivalent JSON Schema dictionary representation, handling various data types and their validations.
