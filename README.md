# 552-Replication

This repository contains a replication of Table 3 (fivefold LATE) from Chernozhukov et al. (2018), *Double/Debiased Machine Learning for Treatment and Structural Parameters*.

The final output of the project is:

- `paper/paper.pdf`

## Repository Structure

- `simple.py`: main replication script used by the `Makefile`
- `Makefile`: builds the table outputs and compiles the paper PDF
- `run_all.sh`: convenience wrapper that runs `make clean` and `make`
- `input/sipp1991.dta`: local copy of the 401(k) application data
- `output/tables/`: generated replication tables
- `paper/paper.tex`: LaTeX source for the write-up
- `paper/paper.pdf`: final compiled paper

## Replication Process

The build pipeline is:

1. `simple.py` loads the 401(k) application data.
2. It estimates the Table 3 fivefold IIVM/DML specification for the 401(k) LATE problem.
3. It writes the generated table files to `output/tables/`.
4. `paper/paper.tex` inputs `output/tables/main_result.tex`.
5. LaTeX compiles the final paper as `paper/paper.pdf`.

`simple.py` uses the same core setup as the paper:

- Outcome: `net_tfa`
- Treatment: `p401`
- Instrument: `e401`
- Controls: `age`, `inc`, `educ`, `fsize`, `marr`, `twoearn`, `db`, `pira`, `hown`
- Estimator: DoubleML IIVM with 5-fold cross-fitting
- Repetitions: 100 repeated sample splits

The script produces these generated outputs:

- `output/tables/table3_5fold_full_results.csv`
- `output/tables/table3_5fold_theta_by_rep.csv`
- `output/tables/table3_5fold_se_by_rep.csv`
- `output/tables/main_result.tex`

## Requirements

You need:

- Python 3
- `make`
- `pdflatex`
- `bibtex`

Python packages used by the main workflow:

- `doubleml`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

Example install:

```bash
pip install doubleml pandas numpy scikit-learn xgboost
```

## How to Run

From the repository root:

```bash
make
```

This does two things:

1. Runs `python3 simple.py`
2. Compiles `paper/paper.tex` into `paper/paper.pdf`

If you want a fully fresh rebuild:

```bash
make clean
make
```

Or use the wrapper script:

```bash
./run_all.sh
```

## Makefile Targets

- `make`: build the table outputs and final PDF
- `make clean`: remove generated table files and LaTeX build artifacts

The default target is:

```make
all: paper/paper.pdf
```

## Data Notes

`simple.py` first tries to load the 401(k) data through `doubleml.datasets.fetch_401K()`. If that download is unavailable, it falls back to the local file:

```text
input/sipp1991.dta
```

That means the repository can still run as long as the included Stata file is present.

## Final Output

After a successful run, the finished paper is here:

```text
paper/paper.pdf
```

## Runtime Note

The full replication is extremely time intensive because `simple.py` runs 100 repeated 5-fold DML estimations across multiple learners. Run it with some caution and expect a long execution time. A full run is estimated to take roughly 10 to 15 hours.
