from typing import Any
from pydantic import BaseModel as PydanticBaseModel


class ParametersBase(PydanticBaseModel):
    def to_kwargs(self) -> dict[str, Any]:
        """파라미터를 딕셔너리로 변환 (param_type 제외)"""
        return self.model_dump(exclude={"param_type", "model_config"})
