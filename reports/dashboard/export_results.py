"""Ejecuta las celdas del notebook 3.0 (sin guardar el modelo) y exporta resultados para el artefacto."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde, ks_2samp

proj = Path(sys.argv[1]); out = Path(sys.argv[2])
nb = json.load(open(proj / "notebooks/3.0-pb-entrenamiento-modelos.ipynb"))
ns = {}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    s = "".join(c["source"])
    if "joblib.dump" in s: continue          # no sobrescribir models/
    if "mlflow.start_run" in s or "register_model" in s: continue  # no registrar runs ni versiones en MLflow
    exec(compile(s, "nb3", "exec"), ns)

busquedas, torneo, sobreajuste = ns["busquedas"], ns["torneo"], ns["sobreajuste"]
X_val, y_val, y_train = ns["X_val"], ns["y_val"], ns["y_train"]
MODEL_FEATURES, RANGO_TRAIN = ns["MODEL_FEATURES"], ns["RANGO_TRAIN"]

test = pd.read_csv(proj / "data/raw/test.csv", index_col="Id")
inferencia = pd.read_csv(proj / "data/processed/predictions.csv", index_col="Id")["SalePrice"]
y_train_real = np.expm1(y_train); y_val_real = np.expm1(y_val)
max_train = float(y_train_real.max())

grid = np.linspace(0, 800_000, 241)
def kde(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(x) for x in gaussian_kde(v)(grid)]
def q(v):
    v = np.asarray(v, float)
    return {k: float(f(v)) for k, f in [("min", np.min), ("p05", lambda a: np.percentile(a, 5)), ("mediana", np.median), ("p95", lambda a: np.percentile(a, 95)), ("max", np.max)]}

best_nrmse = float(torneo["nrmse_val"].min())
models = []
for name in torneo.index:
    s = busquedas[name]; bi = s.best_index_
    pred_test = np.expm1(s.best_estimator_.predict(test[MODEL_FEATURES]))
    pred_val = np.expm1(s.best_estimator_.predict(X_val))
    folds = [float(-s.cv_results_[f"split{k}_test_score"][bi]) for k in range(10)]
    folds_train = [float(-s.cv_results_[f"split{k}_train_score"][bi]) for k in range(10)]
    rmse_cv = float(-s.cv_results_["mean_test_score"][bi]); std_cv = float(s.cv_results_["std_test_score"][bi])
    nrmse_val = float(torneo.loc[name, "nrmse_val"])
    ks = float(ks_2samp(pred_test[np.isfinite(pred_test)], y_train_real).statistic)
    fuera = int((~np.isfinite(pred_test) | (pred_test < 0) | (pred_test > 1.5 * max_train)).sum())
    criterios = {
        "nrmse_alto": {"falla": bool(nrmse_val > 2 * best_nrmse), "valor": nrmse_val, "umbral": 2 * best_nrmse},
        "fuera_de_rango": {"falla": bool(fuera > 0), "valor": fuera, "umbral": 0},
        "distribucion_distinta": {"falla": bool(ks > 0.2), "valor": ks, "umbral": 0.2},
        "inestable_cv": {"falla": bool(std_cv > 0.5 * rmse_cv), "valor": std_cv / rmse_cv, "umbral": 0.5},
    }
    configs = []
    for i in range(len(s.cv_results_["params"])):
        configs.append({
            "params": {k.replace("modelo__", ""): (str(v) if not isinstance(v, (int, float)) or isinstance(v, bool) else v) for k, v in s.cv_results_["params"][i].items()},
            "rmse_cv": float(-s.cv_results_["mean_test_score"][i]),
            "std_cv": float(s.cv_results_["std_test_score"][i]),
            "rmse_train": float(-s.cv_results_["mean_train_score"][i]),
            "rank": int(s.cv_results_["rank_test_score"][i]),
            "best": bool(i == bi),
        })
    models.append({
        "nombre": name,
        "estimador": type(s.best_estimator_.named_steps["modelo"]).__name__,
        "best_params": {k.replace("modelo__", ""): str(v) for k, v in s.best_params_.items()},
        "rmse_val": float(torneo.loc[name, "rmse_real_val"]), "nrmse_val": nrmse_val,
        "rmse_cv": rmse_cv, "std_cv": std_cv,
        "rmse_train_cv": float(sobreajuste.loc[name, "rmse_train_cv"]),
        "brecha_pct": float(sobreajuste.loc[name, "brecha_%"]),
        "folds_cv": folds, "folds_train": folds_train,
        "ks_test_vs_train": ks, "n_fuera_rango": fuera,
        "criterios": criterios,
        "descartado": any(c["falla"] for c in criterios.values()),
        "val_pred": [round(float(x)) for x in pred_val],
        "test_kde": kde(pred_test), "test_cuantiles": q(pred_test),
        "test_igual_inferencia": bool(np.allclose(pred_test, inferencia.loc[test.index].values)),
        "configs": configs,
    })

data = {
    "generado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(test)), "n_folds": 10,
    "rango_train": float(RANGO_TRAIN), "max_train": max_train, "best_nrmse": best_nrmse,
    "y_val": [round(float(x)) for x in y_val_real],
    "grid": [float(x) for x in grid],
    "train_kde": kde(y_train_real), "train_cuantiles": q(y_train_real),
    "inferencia_kde": kde(inferencia.values), "inferencia_cuantiles": q(inferencia.values),
    "modelos": models,
}
json.dump(data, open(out, "w"), ensure_ascii=False)

ddir = out.parent / "dash_data"; ddir.mkdir(exist_ok=True)
etiquetas = {"nrmse_alto": "NRMSE val > 2x el mejor", "fuera_de_rango": "Predicciones fuera de rango",
             "distribucion_distinta": "KS vs train > 0.2", "inestable_cv": "Desv. CV > 50% de la media"}
modelos_rows, configs_rows, folds_rows, val_rows, dens_rows = [], [], [], [], []
for m in models:
    c = m["criterios"]; tq = m["test_cuantiles"]
    modelos_rows.append({
        "modelo": m["nombre"], "estimador": m["estimador"],
        "mejores_hiperparametros": ", ".join(f"{k}={v}" for k, v in m["best_params"].items()) or "sin hiperparámetros",
        "n_configuraciones": len(m["configs"]),
        "rmse_val": round(m["rmse_val"], 1), "nrmse_val": round(m["nrmse_val"], 5),
        "rmse_cv": round(m["rmse_cv"], 1), "std_cv": round(m["std_cv"], 1), "cv_relativa": round(m["std_cv"] / m["rmse_cv"], 4),
        "rmse_train_cv": round(m["rmse_train_cv"], 1), "brecha_pct": round(m["brecha_pct"], 1),
        "ks_test_vs_train": round(m["ks_test_vs_train"], 4), "n_fuera_rango": m["n_fuera_rango"],
        "falla_nrmse": c["nrmse_alto"]["falla"], "falla_rango": c["fuera_de_rango"]["falla"],
        "falla_ks": c["distribucion_distinta"]["falla"], "falla_cv": c["inestable_cv"]["falla"],
        "descartado": m["descartado"],
        "motivos": "; ".join(etiquetas[k] for k, v in c.items() if v["falla"]) or "",
        "estado": "Descartado" if m["descartado"] else "Válido",
        "test_min": round(tq["min"]), "test_p05": round(tq["p05"]), "test_mediana": round(tq["mediana"]),
        "test_p95": round(tq["p95"]), "test_max": round(tq["max"]),
        "coincide_con_inferencia": m["test_igual_inferencia"],
    })
    for cfg in m["configs"]:
        configs_rows.append({"modelo": m["nombre"],
            "configuracion": ", ".join(f"{k}={v}" for k, v in cfg["params"].items()) or "sin hiperparámetros",
            "rmse_cv": round(cfg["rmse_cv"], 1), "std_cv": round(cfg["std_cv"], 1),
            "rmse_train": round(cfg["rmse_train"], 1), "ranking": cfg["rank"], "seleccionada": cfg["best"]})
    for k, (te, tr) in enumerate(zip(m["folds_cv"], m["folds_train"]), 1):
        folds_rows.append({"modelo": m["nombre"], "fold": k, "rmse_validacion_fold": round(te, 1), "rmse_entrenamiento_fold": round(tr, 1)})
    for yr, yp in zip(data["y_val"], m["val_pred"]):
        val_rows.append({"modelo": m["nombre"], "real": yr, "predicho": yp})
    for x, d in zip(grid, m["test_kde"]):
        dens_rows.append({"serie": m["nombre"], "precio": round(float(x)), "densidad": d})
for serie, key in [("SalePrice train (real)", "train_kde"), ("Pipeline de inferencia", "inferencia_kde")]:
    for x, d in zip(grid, data[key]):
        dens_rows.append({"serie": serie, "precio": round(float(x)), "densidad": d})
ref = [{"n_train": data["n_train"], "n_val": data["n_val"], "n_test": data["n_test"], "n_folds": 10,
        "rango_train": round(data["rango_train"]), "max_train": round(data["max_train"]),
        "limite_rango": round(1.5 * data["max_train"]), "mejor_nrmse": round(best_nrmse, 5),
        "umbral_nrmse": round(2 * best_nrmse, 5), "umbral_ks": 0.2, "umbral_cv": 0.5,
        "train_mediana": round(data["train_cuantiles"]["mediana"]), "inferencia_mediana": round(data["inferencia_cuantiles"]["mediana"]),
        "generado": data["generado"]}]
for name, rows in [("modelos", modelos_rows), ("configuraciones_cv", configs_rows), ("folds_cv", folds_rows),
                   ("predicciones_validacion", val_rows), ("densidades_test", dens_rows), ("referencia", ref)]:
    json.dump(rows, open(ddir / f"{name}.json", "w"), ensure_ascii=False)
    print(name, len(rows), (ddir / f"{name}.json").stat().st_size)
for m in models:
    print(f"{m['nombre']:<18} rmse_val={m['rmse_val']:>10,.0f} nrmse={m['nrmse_val']:.4f} ks={m['ks_test_vs_train']:.3f} fuera={m['n_fuera_rango']} cv_cv={m['std_cv']/m['rmse_cv']:.2f} igual_inf={m['test_igual_inferencia']} -> {'DESCARTADO ' + str([k for k,c in m['criterios'].items() if c['falla']]) if m['descartado'] else 'OK'}")
print("tamaño json:", out.stat().st_size)
