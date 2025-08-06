# A tertiary systematic mapping on IoT middleware and its application of Artificial Intelligence

Data obtained during the mapping execution is available in the `data` folder. The scripts used to
read these data files, plot graphs, and report metrics are available in the `scripts` folder.

## Data

- `data/reviews.bib` contains all unique reviews that were retrieved with the search string,
annotated with `true` or `false` for each inclusion and exclusion criterion.
- `data/selected-reviews.bib` contains only reviews selected by the inclusion/exclusion criteria.
- `data/selected-reviews.csv` contains additional data collected about each selected review.
- `data/detailed-aspects.csv` contains a detailed mapping of specific topics among the aspects each
selected review that were covered by selected reviews.
- `data/middlewares.bib` contains bibliographical details about each middleware instance cited by
selected reviews.
- `data/middlewares.csv` contains additional data collected about each middleware instance cited by
selected reviews

## Scripts

In order to run the scripts, install the `requirements.txt` in a Python virtual environemnt,
activate it, then run:

```bash
$ python -m scripts --help
```

The help message explains further details about available commands.
