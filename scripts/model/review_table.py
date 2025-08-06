from __future__ import annotations
from operator import itemgetter as at
from pathlib import Path
from scripts.model.mw_table import MiddlewaresTable, Col as MwCol
import bibtexparser as btp
import pandas as pd
import re


class Col:
    id = "id"
    ref = "study"
    year = "year"
    topic = "topic"
    methods = "methodology"
    spec = "mw_specific"
    range = "range"
    aspects = "aspects"
    country = "country"
    range_start = "range_start"
    range_end = "range_end"
    about_ai = "about_ai"
    is_systematic = "is_systematic"


class ReviewTable:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def read(cls, path: Path, bib: Path, sort_by: list[str]) -> ReviewTable:
        sort_by_cols, asc = cls._parse_sort_by(sort_by)
        lib = btp.parse_file(bib)
        df = (
            pd.read_csv(path, header=0)
            .assign(year=lambda df: df[Col.ref].apply(lambda r: cls.ref_year(lib, r)))
            .sort_values(list(sort_by_cols), ascending=asc)
            .assign(id=lambda df: [f"S{i + 1:02d}" for i in range(len(df))])
        )
        return cls(df)

    @staticmethod
    def _parse_sort_by(sort_by: list[str]) -> tuple[list[str], list[bool]]:
        ms = (
            re.match(r"(\w+)(?:\s+(asc|desc))?", s, flags=re.IGNORECASE)
            for s in sort_by
        )
        keys = ((m.group(1), (m.group(2) or "asc").lower() == "asc") for m in ms if m)
        return tuple(zip(*keys))

    @staticmethod
    def ref_year(lib: btp.Library, ref: str) -> int | None:
        try:
            return int(lib.entries_dict[ref].fields_dict[Col.year].value)
        except KeyError:
            return None

    def with_aspects(self) -> ReviewTable:
        df = self.df.copy()
        df[Col.aspects] = self.df[Col.aspects].str.split()
        df[Col.about_ai] = self.df[Col.aspects].apply(lambda aspects: "ai" in aspects)
        return ReviewTable(df)

    def with_systematic(self) -> ReviewTable:
        return ReviewTable(
            self.df.assign(**{Col.is_systematic: self.df[Col.methods] != "adhoc"})
        )

    def with_range(self, mws: MiddlewaresTable) -> ReviewTable:
        def get_range(year_range: str) -> tuple[int, int] | None:
            if m := re.match(r"^(\d+)~(\d+)$", year_range):
                return tuple(map(int, m.groups()))
            return None

        referenced_range_df = (
            mws.df.explode(MwCol.rev_list)
            .groupby(MwCol.rev_list)[MwCol.latest_update]
            .agg(["min", "max"])
            .reset_index()
            .assign(**{Col.ref: lambda df: df[MwCol.rev_list]})
            .assign(
                ymin=lambda df: df["min"],
                ymax=lambda df: df["max"],
            )[[Col.ref, "ymin", "ymax"]]
        )

        idf = (
            self.df.merge(referenced_range_df, on=Col.ref, how="left")
            .fillna(-1)
            .assign(
                ref_range=lambda df: pd.Series(
                    zip(df["ymin"].astype(int), df["ymax"].astype(int))
                )
            )
        )

        df = self.df.copy()
        df["t_range"] = idf[Col.range].apply(get_range).combine_first(idf["ref_range"])
        df = df.assign(
            range_start=lambda df: df["t_range"].apply(at(0)),
            range_end=lambda df: df["t_range"].apply(at(1)),
        ).drop(["t_range"], axis=1)
        return ReviewTable(df)
