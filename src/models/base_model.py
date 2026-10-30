from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        pass

    @property
    def name(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}')"
