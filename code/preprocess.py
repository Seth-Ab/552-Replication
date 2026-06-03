"""Preprocess the 401(k) application data for the Table 3 replication."""

from pathlib import Path
from urllib.error import URLError

import pandas as pd
from doubleml.datasets import fetch_401K


RAW_FALLBACK = Path("input/sipp1991.dta")
OUTFILE = Path("temp/clean_data.csv")
VARS = [
    "net_tfa",
    "p401",
    "e401",
    "age",
    "inc",
    "educ",
    "fsize",
    "marr",
    "twoearn",
    "db",
    "pira",
    "hown",
]


def load_data() -> pd.DataFrame:
    if RAW_FALLBACK.exists():
        return pd.read_stata(RAW_FALLBACK)

    try:
        return fetch_401K(return_type="DataFrame")
    except URLError:
        return pd.read_stata(RAW_FALLBACK)


def main() -> None:
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    data = load_data()
    clean = data[VARS].dropna().copy()
    clean.to_csv(OUTFILE, index=False)

    print("Preprocessing complete")
    print("N observations:", clean.shape[0])
    print("Variables:", ", ".join(clean.columns))
    print("Age range:", int(clean["age"].min()), "to", int(clean["age"].max()))
    print("Mean net_tfa:", round(clean["net_tfa"].mean(), 2))
    print("Mean p401:", round(clean["p401"].mean(), 3))
    print("Mean e401:", round(clean["e401"].mean(), 3))
    print("Saved to:", OUTFILE)


if __name__ == "__main__":
    main()
