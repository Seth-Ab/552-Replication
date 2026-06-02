"""Basic fivefold replication of Table 3 from the 401(k) application.

This version keeps the code short and readable. It estimates the LATE of
401(k) participation (`p401`) on net financial assets (`net_tfa`) using
401(k) eligibility (`e401`) as the instrument.
"""

from urllib.error import URLError

import doubleml as dml
import pandas as pd
from doubleml.datasets import fetch_401K
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


SEED = 123
N_FOLDS = 5


def load_data() -> pd.DataFrame:
    try:
        return fetch_401K(return_type="DataFrame")
    except URLError:
        return pd.read_stata("data/sipp1991.dta")


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


def main():
    data = load_data()

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

    print("Basic fivefold Table 3 replication")
    print(results_df.round(3))


if __name__ == "__main__":
    main()
