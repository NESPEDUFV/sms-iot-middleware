from collections.abc import Iterable
from datetime import date
from itertools import repeat
from pathlib import Path
from typing import Annotated

import pandas as pd
from typer import Argument, Context, Option, Typer

from scripts.model.labels import ASPECT_LABELS, METHOD_LABELS
from scripts.model.mw_table import Col as MwCol
from scripts.model.mw_table import MiddlewaresTable
from scripts.model.review_table import Col, ReviewTable

app = Typer()


@app.callback()
def main(
    ctx: Context,
    sort_by: Annotated[list[str], Option(help="Sorting key column")] = ["study"],
):
    """Create LaTeX tables and print to stdout."""
    ctx.ensure_object(dict)
    ctx.obj |= {"sort_by": sort_by}


@app.command()
def studies(
    ctx: Context,
    path: Annotated[
        Path, Argument(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    caption: Annotated[
        str, Option(help="Caption to place on the table")
    ] = "Selected secondary studies.",
    label: Annotated[
        str, Option(help="Label to place on the table")
    ] = "tab:selected-studies",
):
    """Create the study overview table."""

    columns = ["id", Col.ref, "year", Col.methods, Col.spec]
    col_names = [
        "ID",
        "Reference",
        "Year",
        # "Topic",
        "Methodology",
        "Specific",
    ]
    aligns = ["c", "c", "r", "l", "c"]
    format_col = {
        Col.ref: lambda s: rf"\cite{{{s}}}",
        # Col.topic: topic_labels.get,
        Col.methods: lambda x: METHOD_LABELS[x].short,
        Col.spec: lambda b: "\\checkmark" if b else "",
    }

    df = ReviewTable.read(path, bib, ctx.obj["sort_by"]).df.assign(
        **{
            col: lambda df, col=col: df[col].apply(format_col.get(col, lambda x: x))
            for col in columns
        }
    )

    def study_rows(df: pd.DataFrame, columns: list[str]) -> Iterable[str]:
        def row_str(row: Iterable) -> str:
            return " & ".join(map(str, row))

        return map(row_str, df[columns].itertuples(index=False))

    header = " & ".join(rf"\textbf{{{col}}}" for col in col_names)
    studies = " \\\\\n        ".join(study_rows(df, columns))
    table = rf"""
\begin{{table}}[hbtp]
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{ {" ".join(aligns)} }}
        \toprule
        {header} \\
        \midrule
        {studies} \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
""".strip()
    print(table)


@app.command()
def aspects(
    ctx: Context,
    path: Annotated[
        Path, Argument(help="Path to csv file with review data")
    ] = "data/selected-reviews.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with review data")
    ] = "data/selected-reviews.bib",
    caption: Annotated[
        str, Option(help="Caption to place on the table")
    ] = "Summary of middleware aspects discussed in literature.",
    label: Annotated[str, Option(help="Label to place on the table")] = "tab:aspects",
):
    """Create the aspects overview table."""
    labels = {k: v.short for k, v in ASPECT_LABELS.items()}

    df = (
        ReviewTable.read(path, bib, ctx.obj["sort_by"])
        .with_aspects()
        .df.assign(
            **{
                key: lambda df, key=key: df[Col.aspects].apply(lambda a: key in a)
                for key in labels.keys()
            }
        )
    )
    count_df = df[labels.keys()].sum()
    totals = [str(count_df[key]) for key in labels.keys()]
    totals_df = pd.DataFrame(
        zip(labels.keys(), map(int, totals)), columns=("aspect", "count")
    ).sort_values("count", ascending=False)
    print(totals_df["count"].agg(["mean", "median"]))
    print(totals_df)

    def format_rows(df: pd.DataFrame) -> Iterable[str]:
        def row_str(row: tuple) -> str:
            return " & ".join([row[0]] + ["\\checkmark" if v else "" for v in row[1:]])

        return map(row_str, df[["id"] + list(labels.keys())].itertuples(index=False))

    col_names = (rf"\rot{{\textbf{{{v}}}}}" for v in labels.values())
    studies = " \\\\\n        \\hline\n        ".join(format_rows(df))
    table = rf"""
\begin{{table}}
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{ c {" ".join(repeat("c", len(labels)))} }}
        \toprule
        \textbf{{ID}} & {" & ".join(col_names)} \\
        \midrule
        {studies} \\
        \midrule
        Total & {" & ".join(totals)} \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
""".strip()
    print(table)


@app.command()
def middlewares(
    ctx: Context,
    path: Annotated[
        Path, Option(help="Path to csv file with middleware data")
    ] = "data/middlewares.csv",
    bib: Annotated[
        Path, Option(help="Path to bib file with middleware data")
    ] = "data/middlewares.bib",
    caption: Annotated[
        str, Option(help="Caption to place on the table")
    ] = "Top {} middlewares referenced in analyzed reviews, ordered by the number of references.",
    label: Annotated[
        str, Option(help="Label to place on the table")
    ] = "tab:middlewares",
    n: Annotated[int, Option(help="Top `n` middlewares to show.")] = 10,
    time_window: Annotated[
        int, Option(help="Time window in years to analyze recency of middlewares.")
    ] = 10,
):
    """Create the middleware listing table."""
    df = MiddlewaresTable.read(path, bib).by_instance().df
    print(f"Total middlewares mentioned: {len(df)}")
    df = df[df[MwCol.is_iot_middleware]]
    df[MwCol.latest_year] = df[MwCol.latest_year].astype(int)

    n_total = len(df)
    min_year = date.today().year - time_window
    n_in_window = len(df[df[MwCol.latest_year] >= min_year])
    print(f"Filtered middlewares: {n_total}")
    print(f"Middlewares in time window: {n_in_window} ({n_in_window / n_total * 100:.1f}%)")

    mw_rank = (
        df.sort_values(by=MwCol.n_reviews, ascending=False)
        .assign(
            review_cite_list=lambda df: df[MwCol.reviews].apply(
                lambda rs: rf"\cite{{{', '.join(sorted(rs))}}}"
            ),
        )
        .head(n)
        .reset_index(drop=True)
    )
    mw_rank.index += 1
    aligns = ["l", "r", "r", "p{3.7cm}"]
    columns = [MwCol.label, MwCol.latest_year, MwCol.n_reviews, "review_cite_list"]
    col_names = (rf"\textbf{{{v}}}" for v in ["Middleware", "Year", "", "References"])

    def format_rows(df: pd.DataFrame) -> Iterable[str]:
        return (" & ".join(str(x) for x in row) for row in df[columns].itertuples())

    rows = " \\\\\n        \\hline\n        ".join(format_rows(mw_rank))

    table = rf"""
\begin{{table}}[hbtp]
    \caption{{{caption.format(n)}}}
    \label{{{label}}}
    \centering
    \begin{{tabular}}{{ r {" ".join(aligns)} }}
        \toprule
         & {" & ".join(col_names)} \\
        \midrule
        {rows} \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
""".strip()
    print(table)
