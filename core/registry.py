import logging
from typing import Callable, TypeVar

from models.base_model import BaseModel

logger = logging.getLogger("Registry")

T = TypeVar("T", bound=BaseModel)


class ModelRegistry:
    registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        def inner(model: type[T]) -> type[T]:
            assert name not in cls.registry.keys(), f"Cannot add {name} to registry"
            cls.registry[name] = model
            return model

        return inner

    @classmethod
    def get_model(cls, name: str) -> type[BaseModel]:
        model = cls.registry.get(name)

        if model is None:
            raise ValueError(f"Model {name} is not registered.")

        return model
