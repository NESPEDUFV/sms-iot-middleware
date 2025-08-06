from typer import Context, Typer

from scripts.plot import app as plot_app
from scripts.tables import app as table_app

app = Typer()
app.add_typer(plot_app, name="plot")
app.add_typer(table_app, name="tables")


@app.callback()
def main(ctx: Context):
    """Tools to help the research execution and reporting."""
    ctx.ensure_object(dict)


if __name__ == "__main__":
    app()
