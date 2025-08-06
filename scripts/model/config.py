from __future__ import annotations
from pydantic import BaseModel


class DataConfig(BaseModel):
    key: str | list[str]
    label: str | None = None
    extra: bool = False


class Config(BaseModel):
    data: list[DataConfig]
    ic: int
    ec: int

    @classmethod
    def from_json_file(cls, path: str) -> Config:
        with open(path) as f:
            return Config.model_validate_json(f.read())
