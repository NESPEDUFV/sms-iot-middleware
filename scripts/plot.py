from __future__ import annotations

import operator
from collections.abc import Callable
from enum import Enum
from functools import reduce
from itertools import product, repeat
from operator import itemgetter as at
from pathlib import Path
from types import TracebackType
from typing import Annotated, Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fsspec
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
from typer import Argument, Context, Option, Typer

from scripts.model.aspects_table import Aspect, AspectsTable
from scripts.model.aspects_table import Col as AspCol
from scripts.model.labels import METHOD_LABELS, TOPIC_LABELS
from scripts.model.mw_table import Col as MwCol
from scripts.model.mw_table import MiddlewaresTable
from scripts.model.review_table import Col, ReviewTable


class Plot:
    fig: Figure | None
    ax: Axes | None

    def __init__(
        self,
        out: Path,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (6, 4),
        dpi: int = 200,
    ):
        self.out = out
        self.figsize = figsize
        self.nrows = nrows
        self.ncols = ncols
        self.dpi = dpi

    def __enter__(self) -> tuple[Figure, Axes | list[Axes]]:
        fig, ax = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=self.figsize, dpi=self.dpi
        )
        self.fig = fig
        self.ax = ax
        return fig, ax

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if exc_val:
            return
        match (self.nrows, self.ncols):
            case (1, 1):
                self.ax.grid(alpha=0.3)
            case (1, _) | (_, 1):
                for ax in self.ax:
                    ax.grid(alpha=0.3)
            case _:
                for axs in self.ax:
                    for ax in axs:
                        ax.grid(alpha=0.3)
        plt.savefig(self.out, bbox_inches="tight")
        plt.clf()


app = Typer()


@app.callback()
def main(
    ctx: Context,
    out: Annotated[Path, Option(help="Output folder path")] = "images",
    fonts: Annotated[
        list[str], Option("--font", help="Font faces to use in plots")
    ] = [],
    sort_by: Annotated[list[str], Option(help="Sorting key column")] = ["study"],
):
    """Plot figures for the paper."""
    ctx.ensure_object(dict)
    sns.set_theme("paper")
    if fonts:
        plt.rcParams["font.family"] = fonts
    ctx.obj |= {"sort_by": sort_by, "out": out}


@app.command()
def all(
    ctx: Context,
):
    """Plot all figures with default settings."""
    studies_overview(ctx, group_by=OverviewGroup.AI_SPEC)
    countries_map(ctx)
    time_range(ctx)
    aspects(ctx, Aspect.APPROACH)
    aspects(ctx, Aspect.REQUIREMENT)
    aspects(ctx, Aspect.LINK_PROTOCOL)
    aspects(ctx, Aspect.APP_PROTOCOL)
    methods_spec(ctx)


class OverviewGroup(str, Enum):
    TOPIC = Col.topic
    SPEC = Col.spec
    AI_SPEC = "ai_spec"

    def col(self) -> str:
        return self.value

    def format(self, df: pd.DataFrame) -> pd.Series:
        match self:
            case OverviewGroup.TOPIC:
                return df[Col.topic].apply(lambda x: TOPIC_LABELS[x].short)
            case OverviewGroup.SPEC:
                return df[Col.spec].apply(lambda x: "Yes" if x else "No")
            case OverviewGroup.AI_SPEC:
                return [
                    "Specific and addresses AI"
                    if spec and about_ai
                    else "Specific"
                    if spec
                    else "Addresses AI"
                    if about_ai
                    else "None"
                    for spec, about_ai in zip(df[Col.spec], df[Col.about_ai])
                ]

    def legend(self):
        match self:
            case OverviewGroup.TOPIC:
                plt.legend(
                    loc="upper left",  # loc="lower center",
                    bbox_to_anchor=(1, 1),  # bbox_to_anchor=(0.5, 1),
                    # ncols=len(topic_short_labels) // 2,
                )
            case OverviewGroup.SPEC:
                plt.legend(
                    title="Specific to\nmiddleware?",
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                )
            case OverviewGroup.AI_SPEC:
                plt.legend(
                    title="Specific to IoT middleware\nor addresses AI?",
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                )

    def label_order_key(self) -> Callable[[str], int] | None:
        match self:
            case OverviewGroup.AI_SPEC:
                labels = (
                    "None",
                    "Specific",
                    "Addresses AI",
                    "Specific and addresses AI",
                )
                return labels.index
            case _:
                return None

    def label_color(self, label: str) -> str | None:
        match self:
            case OverviewGroup.AI_SPEC:
                colors = {
                    "None": "tab:gray",
                    "Addresses AI": "tab:blue",
                    "Specific": "tab:red",
                    "Specific and addresses AI": "tab:purple",
                }
                return colors[label]
            case _:
                return None


@app.command()
def studies_overview(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    group_by: Annotated[
        OverviewGroup, Option(help="Criterion to group studies.")
    ] = OverviewGroup.AI_SPEC,
):
    """Plot overview of studies along time."""
    out: Path = ctx.obj["out"]
    df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).with_aspects().df
    col = group_by.col()

    count: pd.DataFrame = (
        df.assign(**{col: group_by.format}).groupby([col, Col.year])[Col.ref].count()
    )
    groups = sorted(list({g for g, _ in count.index}), key=group_by.label_order_key())
    x = sorted(list({year for _, year in count.index}))

    bottom = np.zeros(len(x))

    with Plot(out / f"studies-overview-{group_by.value.replace('_', '-')}.png"):
        for group in groups:
            y = np.array([count[group].get(year, 0) for year in x])
            plt.bar(
                x, y, 0.5, label=group, bottom=bottom, color=group_by.label_color(group)
            )
            bottom += y

        plt.ylim(0, max(bottom) + 0.5)
        plt.ylabel("Number of studies")
        plt.xlabel("Year")
        group_by.legend()


@app.command()
def countries(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    only_first: Annotated[bool, Option(help="Consider only first authors")] = True,
):
    """Plot nationality of authors."""

    out: Path = ctx.obj["out"]
    df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).df
    countries = df[Col.country].str.split()
    countries = countries.apply(at(0)) if only_first else countries.explode()
    countries = countries.value_counts()
    xticks = list(countries.index)

    with Plot(out / "countries.png") as (_, ax):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.bar(x=xticks, height=countries)
        plt.xticks(xticks, xticks, rotation=45 if only_first else 90)
        plt.ylabel("Number of authors")
        plt.xlabel("Country")


@app.command()
def countries_map(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    only_first: Annotated[bool, Option(help="Consider only first authors")] = True,
):
    """Plot nationality of authors."""

    out: Path = ctx.obj["out"]
    df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).df

    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )

    with fsspec.open(f"simplecache::{url}") as file:
        wmap: gpd.GeoDataFrame = gpd.read_file(file)
        wmap["name"] = wmap["SOVEREIGNT"]
        wmap = wmap.drop(
            [col for col in wmap.columns if col not in {"name", "geometry"}], axis=1
        )

    wmap = wmap[wmap["name"] != "Antarctica"]
    offset_df = pd.read_csv("tables/country-offset.csv")
    countries = df[Col.country].str.split()
    countries = countries.apply(at(0)) if only_first else countries.explode()
    countries_abs = countries.value_counts().reset_index()
    cmap = sns.color_palette("mako", as_cmap=True)
    filtered_wmap = wmap.merge(
        countries_abs,
        left_on="name",
        right_on="country",
        how="right",
    ).merge(offset_df)
    filtered_wmap["color"] = filtered_wmap["count"].apply(
        lambda x: cmap(x / (countries_abs["count"].max() + 0.75))
    )
    filtered_wmap = filtered_wmap[["country", "geometry", "count", "color"]]
    bounds = wmap["geometry"].apply(lambda geo: geo.bounds)
    abs_bounds = {
        "x_min": bounds.apply(at(0)).min(),
        "y_min": bounds.apply(at(1)).min(),
        "x_max": bounds.apply(at(2)).max(),
        "y_max": bounds.apply(at(3)).max(),
    }

    with Plot(out / "countries-map.png", figsize=(7, 6)) as (_, ax):
        wmap.plot(ax=ax, color="lightgrey")
        ax.set_axis_off()
        filtered_wmap.plot(
            color=filtered_wmap["color"],
            ax=ax,
        )
        ax.set_xlim(left=abs_bounds["x_min"], right=abs_bounds["x_max"])
        ax.set_ylim(bottom=abs_bounds["y_min"], top=abs_bounds["y_max"])
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    markersize=row[1] * 3,
                    linestyle="",
                    marker="o",
                    color=row[0],
                    label=f"{row[1]} {row[2]}",
                )
                for row in filtered_wmap[["color", "count", "country"]].itertuples(
                    index=False
                )
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )


class RangePlotMode(str, Enum):
    ERROR = "error"
    BAR = "bar"


@app.command()
def time_range(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    mw: Annotated[
        Path, Option(help="Path to csv file with middleware data")
    ] = "tables/middlewares.csv",
    mw_bib: Annotated[
        Path, Option(help="Path to bib file with middleware data")
    ] = "tables/middlewares.bib",
):
    """Plot time range covered by reviews."""
    out: Path = ctx.obj["out"]
    mw_df = (
        MiddlewaresTable.read(mw, mw_bib)
        .df[[MwCol.cited_by, MwCol.label, MwCol.year]]
        .rename({MwCol.year: "mw_year", MwCol.label: "mw_label"}, axis=1)
    )
    reviews_df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).with_systematic().df
    reviews_df[Col.is_systematic] = np.where(
        reviews_df[Col.is_systematic], "Systematic", "Ad Hoc"
    )
    df = mw_df.merge(reviews_df, left_on=MwCol.cited_by, right_on=Col.ref)

    print("Min year:")
    print(df.groupby(Col.is_systematic)["mw_year"].min())

    with Plot(out / "time-range.png", figsize=(8, 4)) as (_, ax):
        sns.histplot(
            data=df,
            x="mw_year",
            hue=Col.is_systematic,
            multiple="layer",
            fill=True,
            legend=True,
            discrete=True,
            ax=ax,
        )
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.get_legend().set_title("")
        plt.ylabel("Primary study count")
        plt.xlabel("Year")

    diff_df = df.assign(diff=df[Col.year] - df["mw_year"])
    print("Mean time diff:")
    print(diff_df.groupby(Col.is_systematic)["diff"].median())

    with Plot(out / "time-diff.png", figsize=(8, 3)) as (_, ax):
        sns.boxplot(
            data=diff_df,
            x="diff",
            hue=Col.is_systematic,
            hue_order=("Ad Hoc", "Systematic"),
            gap=0.2,
            width=0.8,
        )
        # plt.ylabel("Methodology type")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.get_legend().set_title("")
        plt.xlabel("Time difference (years)")


@app.command()
def aspects(
    ctx: Context,
    aspect: Annotated[Aspect, Argument(help="Which aspect to plot")],
    path: Annotated[
        Path, Option(help="Path to csv file with aspects data")
    ] = "data/middlewares.bib",
    min: Annotated[int, Option(help="Min number of mentions to include an item")] = 1,
    n: Annotated[Optional[int], Option(help="Max number of items to plot")] = None,
):
    """Plot count of detailed aspects."""
    out: Path = ctx.obj["out"]
    df = AspectsTable.read(path).without_alias().filtered(aspect).df
    total = len(df[AspCol.study].unique())
    items = df[AspCol.item].value_counts()
    print(f"Before: {len(items)}")
    items = items[items > min]
    print(f"After: {len(items)}")
    if n:
        items = items.head(n)
    items = items.iloc[::-1]
    ticks = list(items.index)

    with Plot(out / f"aspect-{aspect.slug()}.png") as (_, ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        p = plt.barh(y=ticks, width=items)
        plt.bar_label(
            p,
            labels=[f"{(x / total) * 100:5.1f}%" for x in items],
            padding=-38,
            color="white",
        )
        plt.yticks(ticks, ticks)
        plt.xlabel("Number of mentions")
        plt.ylabel(aspect.label(plural=True).capitalize())


@app.command()
def methods(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    short: Annotated[bool, Option(help="If enabled, labels are shortened.")] = False,
):
    """Plot count of study methodologies"""
    out: Path = ctx.obj["out"]
    df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).df
    labels = {k: v.short if short else v.long for k, v in METHOD_LABELS.items()}
    items = df[Col.methods].apply(labels.get).value_counts()
    ticks = list(items.index)

    with Plot(out / "methods.png", figsize=(8, 6)) as (_, ax):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.bar(x=ticks, height=items)
        plt.xticks(ticks, ticks)
        plt.ylabel("Number of studies")
        plt.xlabel("Methodology")


@app.command()
def methods_spec(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
):
    """Plot count of study methodologies"""
    out: Path = ctx.obj["out"]
    df = (
        ReviewTable.read(path, bib, ctx.obj["sort_by"])
        .with_aspects()
        .with_systematic()
        .df
    )

    with Plot(out / "methods-spec.png") as (_, ax):
        venn3_from_df(
            df,
            cols=(Col.is_systematic, Col.spec, Col.about_ai),
            labels=("Systematic", "Specific to\nIoT middleware", "Addresses AI"),
            ax=ax,
        )


def venn3_from_df(
    df: pd.DataFrame,
    cols: tuple[str, str, str],
    labels: tuple[str, str, str],
    ax: plt.Axes,
    weighted: bool = True,
    colors: tuple[str, str, str] = ("tab:green", "tab:red", "tab:blue"),
) -> None:
    set_counts = {
        "".join(map(str, map(int, bools))): reduce(
            operator.and_, (df[col] == b for col, b in zip(cols, bools))
        ).sum()
        for bools in product(*repeat((False, True), 3))
    }
    venn3(
        subsets=set_counts,
        set_labels=labels,
        set_colors=colors,
        layout_algorithm=DefaultLayoutAlgorithm(
            fixed_subset_sizes=None if weighted else (1,) * 8
        ),
        ax=ax,
    )
