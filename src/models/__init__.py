from .base_model import BaseModel

__all__ = [
    "BaseModel",
    "OpenAIModel",
    "AzureOpenAIModel",
    "HuggingFaceModel",
    "vllmModel",
    "VLLMModel",
]


def __getattr__(name):
    if name == "OpenAIModel":
        from .openai_model import OpenAIModel
        return OpenAIModel
    if name == "AzureOpenAIModel":
        from .azure_openai_model import AzureOpenAIModel
        return AzureOpenAIModel
    if name == "HuggingFaceModel":
        from .huggingface_model import HuggingFaceModel
        return HuggingFaceModel
    if name in ("vllmModel", "VLLMModel"):
        from .vllm_model import vllmModel
        return vllmModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
