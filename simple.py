"""Python Table 3 (fivefold) replication with partner-style outputs.

This script targets Table 3 from Chernozhukov et al. (2018):
- estimand: LATE of 401(k) participation (`p401`)
- instrument: 401(k) eligibility (`e401`)
- estimator class: IIVM / DML with fivefold cross-fitting

It also aligns the output format and the simplified `Ensemble` / `Best`
construction with your partner's R workflow:
- one 100-repetition theta matrix
- one 100-repetition standard-error matrix
- one 3-row summary CSV
- one LaTeX table fragment

`Ensemble` and `Best` follow your partner's simplified logic:
- `Ensemble`: simple average of nuisance predictions across the five base learners
- `Best`: single base learner with the lowest combined out-of-fold nuisance MSE

That keeps the process group-consistent while staying true to Table 3's IIVM
setup rather than switching to the PLR/ATE setup from Table 2.
"""

from __future__ import annotations

from pathlib import Path
import time
from urllib.error import URLError

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.datasets import fetch_401K
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


SEED = 123
N_FOLDS = 5
N_REP = 100
TRIMMING_THRESHOLD = 0.01

BASE_X_COLS = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
BASE_METHODS = ["Lasso", "Reg. Tree", "Forest", "Boosting", "Neural Net"]
ALL_METHODS = BASE_METHODS + ["Ensemble", "Best"]

PAPER_ATE = np.array([8944, 11459, 11764, 11133, 11186, 11173, 11113])
PAPER_SPLIT_SE = np.array([2259, 1717, 1788, 1661, 1795, 1641, 1645])
PAPER_ADJ_SE = np.array([3307, 1786, 1893, 1710, 1890, 1678, 1675])

OUTPUT_DIR = Path("output/tables")


def load_data() -> pd.DataFrame:
    try:
        return fetch_401K(return_type="DataFrame")
    except URLError:
        return pd.read_stata("input/sipp1991.dta")


def make_base_dml_data(data: pd.DataFrame) -> dml.DoubleMLData:
    return dml.DoubleMLData(
        data,
        y_col="net_tfa",
        d_cols="p401",
        z_cols="e401",
        x_cols=BASE_X_COLS,
    )


def make_flex_dml_data(data: pd.DataFrame) -> dml.DoubleMLData:
    # Table 3 follows the same richer lasso feature idea as the paper's 401(k)
    # setup: raw covariates plus second-order terms for the penalized learner.
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_flex = pd.DataFrame(
        poly.fit_transform(data[BASE_X_COLS]),
        columns=poly.get_feature_names_out(BASE_X_COLS),
        index=data.index,
    )
    flex_data = pd.concat([data[["net_tfa", "p401", "e401"]], x_flex], axis=1)
    return dml.DoubleMLData(
        flex_data,
        y_col="net_tfa",
        d_cols="p401",
        z_cols="e401",
    )


def make_smpls(n_obs: int, n_folds: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    shuffled = rng.permutation(n_obs)
    folds = np.array_split(shuffled, n_folds)
    smpls = []
    for test_idx in folds:
        train_idx = np.setdiff1d(shuffled, test_idx, assume_unique=True)
        smpls.append((train_idx, test_idx))
    return smpls


def extract_predictions(dml_obj: dml.DoubleMLIIVM) -> dict[str, np.ndarray]:
    return {name: np.asarray(values).reshape(-1) for name, values in dml_obj.predictions.items()}


def iivm_effect_from_predictions(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    preds: dict[str, np.ndarray],
) -> tuple[float, float]:
    g0 = preds["ml_g0"]
    g1 = preds["ml_g1"]
    m = preds["ml_m"]
    r0 = preds["ml_r0"]
    r1 = preds["ml_r1"]

    psi_b = g1 - g0 + z * (y - g1) / m - (1.0 - z) * (y - g0) / (1.0 - m)
    psi_a = -1.0 * (r1 - r0 + z * (d - r1) / m - (1.0 - z) * (d - r0) / (1.0 - m))
    theta = -np.mean(psi_b) / np.mean(psi_a)
    psi = psi_a * theta + psi_b
    se = np.sqrt(np.mean(psi**2) / (np.mean(psi_a) ** 2) / len(y))
    return float(theta), float(se)


def combined_nuisance_mse(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    preds: dict[str, np.ndarray],
) -> float:
    z0 = z == 0
    z1 = z == 1
    mse_g0 = np.mean((y[z0] - preds["ml_g0"][z0]) ** 2)
    mse_g1 = np.mean((y[z1] - preds["ml_g1"][z1]) ** 2)
    mse_m = np.mean((z - preds["ml_m"]) ** 2)
    mse_r0 = np.mean((d[z0] - preds["ml_r0"][z0]) ** 2)
    mse_r1 = np.mean((d[z1] - preds["ml_r1"][z1]) ** 2)
    return float(mse_g0 + mse_g1 + mse_m + mse_r0 + mse_r1)


def fit_single_rep(
    name: str,
    dml_data: dml.DoubleMLData,
    ml_g,
    ml_m,
    ml_r,
    smpls: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, dict[str, np.ndarray], float]:
    model = dml.DoubleMLIIVM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        ml_r=ml_r,
        n_folds=N_FOLDS,
        n_rep=1,
        trimming_threshold=TRIMMING_THRESHOLD,
        subgroups={"always_takers": False, "never_takers": True},
    )
    model.set_sample_splitting(smpls)
    model.fit(store_predictions=True)

    preds = extract_predictions(model)
    y = dml_data.y.reshape(-1)
    d = dml_data.d.reshape(-1)
    z = dml_data.z.reshape(-1)
    theta, se = iivm_effect_from_predictions(y, d, z, preds)
    mse = combined_nuisance_mse(y, d, z, preds)
    return theta, se, preds, mse


def make_lasso_learners():
    lasso_g = make_pipeline(StandardScaler(), LassoCV(cv=10, max_iter=20000))
    lasso_m = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            cv=10,
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            Cs=np.logspace(-3, 2, 20),
        ),
    )
    lasso_r = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            cv=10,
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            Cs=np.logspace(-3, 2, 20),
        ),
    )
    return lasso_g, lasso_m, lasso_r


def make_tree_learners():
    tree_g = DecisionTreeRegressor(random_state=SEED, min_samples_leaf=20, ccp_alpha=0.0015)
    tree_m = DecisionTreeClassifier(random_state=SEED, min_samples_leaf=40, ccp_alpha=0.0025)
    tree_r = DecisionTreeClassifier(random_state=SEED, min_samples_leaf=20, ccp_alpha=0.05)
    return tree_g, tree_m, tree_r


def make_forest_learners():
    forest_g = RandomForestRegressor(
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    forest_m = RandomForestClassifier(
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    forest_r = RandomForestClassifier(
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    return forest_g, forest_m, forest_r


def make_boost_learners():
    boost_g = XGBRegressor(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.5,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
    )
    boost_m = XGBClassifier(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
    )
    boost_r = XGBClassifier(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
    )
    return boost_g, boost_m, boost_r


def make_neural_net_learners():
    nn_g = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="logistic",
            alpha=0.01,
            max_iter=2000,
            random_state=SEED,
        ),
    )
    nn_m = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="logistic",
            alpha=0.01,
            max_iter=2000,
            random_state=SEED,
        ),
    )
    nn_r = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="logistic",
            alpha=0.01,
            max_iter=2000,
            random_state=SEED,
        ),
    )
    return nn_g, nn_m, nn_r


def summarize_results(theta_mat: pd.DataFrame, se_mat: pd.DataFrame) -> pd.DataFrame:
    med_theta = theta_mat.median(axis=0)
    med_split_se = se_mat.median(axis=0)
    med_adj_se = pd.Series(
        {
            col: np.median(np.sqrt(se_mat[col].to_numpy() ** 2 + (theta_mat[col].to_numpy() - med_theta[col]) ** 2))
            for col in theta_mat.columns
        }
    )

    return pd.DataFrame(
        [med_theta.round(0), med_split_se.round(0), med_adj_se.round(0)],
        index=["LATE (5-fold)", "[Median SE]", "(Adjusted SE)"],
    )


def write_latex_table(result: pd.DataFrame, outpath: Path) -> None:
    def fmt_row(vals) -> str:
        return " & ".join(str(int(v)) for v in vals)

    rep_ate = result.loc["LATE (5-fold)", ALL_METHODS].astype(int).tolist()
    rep_split = result.loc["[Median SE]", ALL_METHODS].astype(int).tolist()
    rep_adj = result.loc["(Adjusted SE)", ALL_METHODS].astype(int).tolist()

    lines = [
        "% This file is auto-generated by simple.py — do not edit by hand.",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Replication of Table~3 (Five-Fold): Estimated LATE of 401(k)",
        "         Participation on Net Financial Assets (IIVM / DML).}",
        "\\label{tab:main}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & Lasso & Reg.\\ Tree & Forest & Boosting & Neural Net & Ensemble & Best \\\\",
        "\\midrule",
        "\\multicolumn{8}{l}{\\textit{Panel A: Paper targets (Chernozhukov et al., 2018)}} \\\\",
        f" LATE       & {fmt_row(PAPER_ATE)} \\\\",
        "            & " + " & ".join(f"[{int(v)}]" for v in PAPER_SPLIT_SE) + " \\\\",
        "            & " + " & ".join(f"({int(v)})" for v in PAPER_ADJ_SE) + " \\\\",
        "\\midrule",
        "\\multicolumn{8}{l}{\\textit{Panel B: This replication}} \\\\",
        f" LATE       & {fmt_row(rep_ate)} \\\\",
        "            & " + " & ".join(f"[{int(v)}]" for v in rep_split) + " \\\\",
        "            & " + " & ".join(f"({int(v)})" for v in rep_adj) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main() -> None:
    total_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()
    print(f"Loaded {data.shape[0]} observations.")
    print(f"Running {len(BASE_METHODS)} learners with {N_REP} repeated {N_FOLDS}-fold splits.")
    print("Using IIVM / LATE for Table 3.\n")

    data_dml_base = make_base_dml_data(data)
    data_dml_flex = make_flex_dml_data(data)
    y = data["net_tfa"].to_numpy()
    d = data["p401"].to_numpy()
    z = data["e401"].to_numpy()

    learner_specs = [
        ("Lasso", data_dml_flex, make_lasso_learners),
        ("Reg. Tree", data_dml_base, make_tree_learners),
        ("Forest", data_dml_base, make_forest_learners),
        ("Boosting", data_dml_base, make_boost_learners),
        ("Neural Net", data_dml_base, make_neural_net_learners),
    ]

    theta_mat = pd.DataFrame(index=range(N_REP), columns=ALL_METHODS, dtype=float)
    se_mat = pd.DataFrame(index=range(N_REP), columns=ALL_METHODS, dtype=float)
    best_methods = []

    rng = np.random.default_rng(SEED)

    for rep in range(N_REP):
        rep_start = time.time()
        smpls = make_smpls(len(data), N_FOLDS, rng)

        rep_preds: dict[str, dict[str, np.ndarray]] = {}
        rep_mse: dict[str, float] = {}

        for method_name, dml_data, learner_factory in learner_specs:
            ml_g, ml_m, ml_r = learner_factory()
            theta, se, preds, mse = fit_single_rep(method_name, dml_data, ml_g, ml_m, ml_r, smpls)
            theta_mat.loc[rep, method_name] = theta
            se_mat.loc[rep, method_name] = se
            rep_preds[method_name] = preds
            rep_mse[method_name] = mse

        ensemble_preds = {
            key: np.mean([rep_preds[method][key] for method in BASE_METHODS], axis=0)
            for key in ["ml_g0", "ml_g1", "ml_m", "ml_r0", "ml_r1"]
        }
        ensemble_theta, ensemble_se = iivm_effect_from_predictions(y, d, z, ensemble_preds)
        theta_mat.loc[rep, "Ensemble"] = ensemble_theta
        se_mat.loc[rep, "Ensemble"] = ensemble_se

        best_method = min(rep_mse, key=rep_mse.get)
        best_methods.append(best_method)
        best_theta, best_se = iivm_effect_from_predictions(y, d, z, rep_preds[best_method])
        theta_mat.loc[rep, "Best"] = best_theta
        se_mat.loc[rep, "Best"] = best_se

        rep_elapsed = time.time() - rep_start
        total_elapsed = time.time() - total_start
        avg_per_rep = total_elapsed / (rep + 1)
        remaining = avg_per_rep * (N_REP - rep - 1)
        if rep < 5 or (rep + 1) % 10 == 0:
            print(
                f"Rep {rep + 1:3d}/{N_REP} | rep time: {rep_elapsed:5.1f}s | "
                f"elapsed: {format_seconds(total_elapsed)} | ETA: {format_seconds(remaining)} | "
                f"best: {best_method}"
            )

    result = summarize_results(theta_mat, se_mat)
    result.to_csv(OUTPUT_DIR / "table3_5fold_full_results.csv")
    theta_mat.to_csv(OUTPUT_DIR / "table3_5fold_theta_by_rep.csv", index=False)
    se_mat.to_csv(OUTPUT_DIR / "table3_5fold_se_by_rep.csv", index=False)
    write_latex_table(result, OUTPUT_DIR / "main_result.tex")

    print("\n=============================================================")
    print(" Table 3  |  LATE  |  5-fold DML")
    print("=============================================================\n")
    formatted = result.copy().astype(int).astype(str)
    formatted.loc["[Median SE]"] = "[" + formatted.loc["[Median SE]"] + "]"
    formatted.loc["(Adjusted SE)"] = "(" + formatted.loc["(Adjusted SE)"] + ")"
    print(formatted)
    print("\n--- Best learner frequency across repetitions ---")
    print(pd.Series(best_methods).value_counts())
    print("\nSaved:")
    print("  output/tables/table3_5fold_full_results.csv")
    print("  output/tables/table3_5fold_theta_by_rep.csv")
    print("  output/tables/table3_5fold_se_by_rep.csv")
    print("  output/tables/main_result.tex")
    print(f"\nTotal runtime: {format_seconds(time.time() - total_start)}")


if __name__ == "__main__":
    main()
