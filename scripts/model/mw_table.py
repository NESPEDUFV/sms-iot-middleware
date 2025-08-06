from __future__ import annotations

import operator
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import TypeVar

import bibtexparser as btp
import numpy as np
import pandas as pd

T = TypeVar("T")
V = TypeVar("V")


class Col:
    label = "label"
    reference = "reference"
    cited_by = "cited_by"
    is_primary = "is_primary"
    is_iot = "iot_related"
    self_categ = "self_nomenclature"
    given_categ = "assigned_nomenclature"

    year = "year"
    latest_year = "latest_year"
    n_reviews = "n_reviews"
    reviews = "reviews"
    is_iot_middleware = "is_iot_middleware"

    is_middleware_self = "is_middleware_self"
    is_middleware_given = "is_middleware_given"


class MiddlewaresTable:
    def __init__(self, df: pd.DataFrame, lib: btp.Library):
        self.df = df
        self.lib = lib

    @classmethod
    def read(cls, path: Path, bib: Path) -> MiddlewaresTable:
        def to_bool(col: str) -> Callable[[pd.DataFrame], pd.Series]:
            return lambda df: df[col].apply(int).apply(bool)

        def to_word_set(col: str) -> Callable[[pd.DataFrame], pd.Series]:
            return lambda df: df[col].str.split().apply(set)

        lib = btp.parse_file(bib)

        def extract_year(ref: str) -> int | None:
            try:
                return int(lib.entries_dict[ref].fields_dict["year"].value)
            except (ValueError, KeyError):
                return None

        df = pd.read_csv(path, header=0, keep_default_na=False).assign(
            **{
                Col.is_primary: to_bool(Col.is_primary),
                Col.is_iot: to_bool(Col.is_iot),
                Col.self_categ: to_word_set(Col.self_categ),
                Col.given_categ: to_word_set(Col.given_categ),
            }
        )
        df[Col.year] = np.where(
            df[Col.is_primary], df[Col.reference].apply(extract_year), None
        )
        return cls(df, lib)

    def by_instance(self) -> MiddlewaresTable:
        df = self.df.groupby(Col.label).agg(
            **{
                Col.latest_year: (Col.year, lambda years: max(filter(lambda y: not pd.isna(y), years), default=None)),
                Col.reviews: (Col.cited_by, set),
                Col.is_middleware_self: (
                    Col.self_categ,
                    lambda c: "middleware" in reduce(operator.or_, c),
                ),
                Col.is_middleware_given: (
                    Col.given_categ,
                    lambda c: "middleware" in reduce(operator.or_, c),
                ),
                Col.is_primary: (Col.is_primary, any),
                Col.is_iot: (Col.is_iot, any),
            }
        )
        df[Col.n_reviews] = df[Col.reviews].apply(len)
        df[Col.is_iot_middleware] = (
            (df[Col.is_middleware_self] | df[Col.is_middleware_given])
            & df[Col.is_primary]
            & df[Col.is_iot]
        )
        df = df.reset_index()
        return MiddlewaresTable(df, self.lib)


# class MiddlewaresTable:
#     def __init__(self, df: pd.DataFrame, lib: btp.Library):
#         self.df = df
#         self.lib = lib

#     @classmethod
#     def read(cls, path: Path, bib: Path) -> MiddlewaresTable:
#         lib = btp.parse_file(bib)
#         df = pd.read_csv(path, header=0, sep=";")

#         def agg_references(
#             agg: Callable[[Iterable[T]], V],
#             cast: Callable[[str], T],
#             field: str,
#             df: pd.DataFrame,
#         ) -> list[V]:
#             all_refs = (row[Col.ref_list] for _, row in df.iterrows())
#             all_entries = (
#                 [lib.entries_dict.get(ref) for ref in refs if ref in lib.entries_dict]
#                 for refs in all_refs
#             )
#             return [
#                 agg(
#                     [
#                         cast(e.fields_dict.get(field).value)
#                         for e in entries
#                         if field in e.fields_dict
#                     ]
#                 )
#                 for entries in all_entries
#             ]

#         def latest_update(df: pd.DataFrame):
#             return agg_references(lambda v: max(v, default=None), int, Col.year, df)

#         def citations(df: pd.DataFrame):
#             return agg_references(sum, lambda x: int(x or 0), Col.citations, df)

#         def qualis(df: pd.DataFrame):
#             return agg_references(
#                 lambda v: min(v, default=9), qualis_value, Col.qualis, df
#             )

#         df = df.assign(
#             **{
#                 Col.ref_list: lambda df: df[Col.refs].str.split(","),
#                 Col.rev_list: lambda df: df[Col.reviews].str.split(","),
#             }
#         ).assign(
#             **{
#                 Col.n_reviews: lambda df: df[Col.rev_list].apply(len),
#                 Col.latest_update: latest_update,
#                 Col.citations: citations,
#                 Col.qualis: qualis,
#             }
#         )
#         return cls(df, lib)
