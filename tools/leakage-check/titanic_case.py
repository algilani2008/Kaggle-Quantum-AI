"""Real-data case study: the Titanic passenger table (carData::TitanicSurvival).

Downloads the CSV from the Rdatasets mirror on GitHub, drops the passenger-name
column straight away (names are not needed), makes a plain random 75/25 split and
audits it. No leak is injected here: this shows what leakcheck reports on a
real, low-dimensional table where identical rows occur naturally.

Run:  python titanic_case.py   (writes results/titanic_report.md and .json)
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from leakcheck import audit

URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/"
       "csv/carData/TitanicSurvival.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    df = pd.read_csv(URL).drop(columns=["rownames"])  # rownames = passenger names
    tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df["survived"])
    rep = audit(tr, te.drop(columns="survived"), target="survived")
    os.makedirs(OUT, exist_ok=True)
    md = rep.to_markdown("Leakage audit: Titanic (random 75/25 split, no injected leak)")
    md += "\nSingle-column out-of-fold AUC against `survived`:\n\n"
    md += "\n".join(f"- {k}: {v:.3f}" for k, v in rep.summary["univariate_scores"].items()) + "\n"
    with open(os.path.join(OUT, "titanic_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT, "titanic_report.json"), "w", encoding="utf-8") as f:
        f.write(rep.to_json())
    print(md)


if __name__ == "__main__":
    main()
