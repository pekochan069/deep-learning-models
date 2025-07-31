from typing import Any
from pydantic import BaseModel as PydanticBaseModel


class ParametersBase(PydanticBaseModel):
    def to_kwargs(self) -> dict[str, Any]:
        """파라미터를 딕셔너리로 변환 (_tag 제외)"""
        return self.model_dump(exclude={"_tag", "model_config"})
