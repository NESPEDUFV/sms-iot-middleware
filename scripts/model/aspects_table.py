from __future__ import annotations
from enum import Enum
from operator import itemgetter as at
from pathlib import Path
import pandas as pd


class Col:
    study = "study"
    aspect = "aspect"
    item = "item"


class Aspect(str, Enum):
    APPROACH = "approach"
    REQUIREMENT = "requirement"
    LINK_PROTOCOL = "link_layer_protocol"
    APP_PROTOCOL = "application_protocol"

    def label(self, plural: bool = True) -> str:
        label = self.value.replace("_", " ")
        if not plural:
            return label
        match self:
            case Aspect.APPROACH:
                return label + "es"
            case _:
                return label + "s"

    def slug(self) -> str:
        return self.value.replace("_", "-")


class AspectsTable:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def read(cls, path: Path):
        return cls(pd.read_csv(path, header=0))

    def without_alias(self) -> AspectsTable:
        self.df[Col.item] = self.df[Col.item].str.split("//").apply(at(0))
        return self

    def filtered(self, aspect: Aspect) -> AspectsTable:
        df = self.df[self.df[Col.aspect] == aspect.value].drop(Col.aspect, axis=1)
        return AspectsTable(df)
