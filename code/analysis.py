"""Simple fivefold Table 3 replication based on the root-level simple.py."""

import os

import doubleml as dml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


SEED = 123
N_FOLDS = 5
INFILE = "temp/clean_data.csv"
OUT_CSV = "output/tables/table3_simple_results.csv"
OUT_TEX = "output/tables/main_result.tex"

PAPER_RESULTS = {
    "lasso": {"coef": 8944, "std_err": 2259},
    "random_forest": {"coef": 11764, "std_err": 1788},
    "regression_tree": {"coef": 11459, "std_err": 1717},
    "boosting": {"coef": 11133, "std_err": 1661},
    "neural_network": {"coef": 11186, "std_err": 1795},
    "ensemble": {"coef": 11173, "std_err": 1641},
    "best": {"coef": 11113, "std_err": 1645},
}


def fit_model(data, name, ml_g, ml_m, ml_r):
    obj_dml_data = dml.DoubleMLData(
        data,
        y_col="net_tfa",
        d_cols="p401",
        z_cols="e401",
        x_cols=["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"],
    )

    dml_obj = dml.DoubleMLIIVM(
        obj_dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        ml_r=ml_r,
        n_folds=N_FOLDS,
        subgroups={"always_takers": False, "never_takers": True},
    )
    dml_obj.fit()

    return {
        "method": name,
        "coef": float(dml_obj.coef[0]),
        "std_err": float(dml_obj.se[0]),
        "t_stat": float(dml_obj.t_stat[0]),
        "p_value": float(dml_obj.pval[0]),
    }


def build_results(data: pd.DataFrame) -> pd.DataFrame:
    results = []

    lasso_g = make_pipeline(StandardScaler(), LassoCV(cv=5))
    lasso_m = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", max_iter=1000),
    )
    lasso_r = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", max_iter=1000),
    )
    results.append(fit_model(data, "lasso", lasso_g, lasso_m, lasso_r))

    forest_g = RandomForestRegressor(n_estimators=200, random_state=SEED)
    forest_m = RandomForestClassifier(n_estimators=200, random_state=SEED)
    forest_r = RandomForestClassifier(n_estimators=200, random_state=SEED)
    results.append(fit_model(data, "random_forest", forest_g, forest_m, forest_r))

    tree_g = DecisionTreeRegressor(max_depth=5, random_state=SEED)
    tree_m = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    tree_r = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    results.append(fit_model(data, "regression_tree", tree_g, tree_m, tree_r))

    boost_g = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=SEED,
    )
    boost_m = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=SEED,
    )
    boost_r = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=SEED,
    )
    results.append(fit_model(data, "boosting", boost_g, boost_m, boost_r))

    nn_g = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), max_iter=500, random_state=SEED),
    )
    nn_m = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=SEED),
    )
    nn_r = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=SEED),
    )
    results.append(fit_model(data, "neural_network", nn_g, nn_m, nn_r))

    results_df = pd.DataFrame(results)

    base_methods = results_df.copy()
    ensemble_row = {
        "method": "ensemble",
        "coef": base_methods["coef"].mean(),
        "std_err": base_methods["std_err"].mean(),
        "t_stat": base_methods["t_stat"].mean(),
        "p_value": base_methods["p_value"].mean(),
    }
    results_df = pd.concat([results_df, pd.DataFrame([ensemble_row])], ignore_index=True)

    best_idx = base_methods["std_err"].idxmin()
    best_row = base_methods.loc[best_idx].copy()
    best_row["method"] = "best"
    results_df = pd.concat([results_df, pd.DataFrame([best_row])], ignore_index=True)

    paper_df = pd.DataFrame.from_dict(PAPER_RESULTS, orient="index").reset_index()
    paper_df = paper_df.rename(columns={"index": "method", "coef": "paper_coef", "std_err": "paper_std_err"})

    merged = results_df.merge(paper_df, on="method", how="left")
    merged["coef_diff"] = merged["coef"] - merged["paper_coef"]
    merged["std_err_diff"] = merged["std_err"] - merged["paper_std_err"]
    return merged


def write_latex_table(results_df: pd.DataFrame) -> None:
    rows = []
    for _, row in results_df.iterrows():
        rows.append(
            (
                row["method"].replace("_", " ").title(),
                f'{row["paper_coef"]:.0f}',
                f'[{row["paper_std_err"]:.0f}]',
                f'{row["coef"]:.0f}',
                f'({row["std_err"]:.0f})',
                f'{row["coef_diff"]:+.0f}',
            )
        )

    body = "\n".join(
        f"{method} & {paper_coef} & {paper_se} & {rep_coef} & {rep_se} & {diff} \\\\"
        for method, paper_coef, paper_se, rep_coef, rep_se, diff in rows
    )

    latex = rf"""\begin{{table}}[h!]
\centering
\caption{{Table 3 fivefold LATE: paper targets versus simple Python replication}}
\label{{tab:main}}
\begin{{tabular}}{{lccccc}}
\toprule
Method & Paper LATE & Paper [SE] & Simple LATE & Simple (SE) & Difference \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\begin{{minipage}}{{0.92\textwidth}}
\small
Notes: Paper targets are the fivefold LATE row from Table 3 of Chernozhukov et al. (2018). The
simple replication uses one fivefold split from \texttt{{simple.py}}, not the paper's 100-split median
aggregation. The simple script's Ensemble and Best rows are simplified constructions and should
not be interpreted as exact replicas of the published definitions.
\end{{minipage}}
\end{{table}}
"""

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)


def main() -> None:
    os.makedirs("output/tables", exist_ok=True)
    data = pd.read_csv(INFILE)
    results_df = build_results(data)
    results_df.to_csv(OUT_CSV, index=False)
    write_latex_table(results_df)

    print("Simple fivefold Table 3 replication")
    print(results_df[["method", "paper_coef", "coef", "coef_diff", "paper_std_err", "std_err"]].round(3))


if __name__ == "__main__":
    main()
