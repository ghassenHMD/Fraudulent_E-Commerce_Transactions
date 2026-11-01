"""
Prépare le déploiement (à lancer après chaque nouvel export du notebook) :

    python make_deploy.py

  1. artifacts_api.joblib : artefact allégé pour l'API (quelques centaines de Ko, sans données)
  2. requirements.txt / requirements-api.txt : dépendances épinglées
     (lightgbm et scikit-learn aux versions du notebook, le reste aux versions installées ici)
  3. .python-version et PYTHON_VERSION de render.yaml
  4. contrôle de parité sur l'artefact allégé
"""
import importlib.metadata as md
import os
import re
import sys

import joblib
import pandas as pd

import fraud_core as fc

SRC, API_OUT = "artifacts.joblib", "artifacts_api.joblib"
KEEP = ["feature_cols", "cat_levels", "age_median", "model_name", "model", "threshold",
        "c_alert", "capacity_per_day", "comparison", "parity", "versions"]


def local(pkg: str):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def plain(obj):
    """DataFrames sans dtype « string » (selon la version de pandas, il dépendrait de pyarrow au rechargement)."""
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        out.index, out.columns = out.index.astype(object), out.columns.astype(object)
        for c in out.columns:
            if pd.api.types.is_string_dtype(out[c]) and not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype(object)
        return out
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    return obj


art = joblib.load(SRC)
joblib.dump(plain({k: art[k] for k in KEEP}), API_OUT, compress=3)
print(f"{API_OUT} : {os.path.getsize(API_OUT) / 1e3:.0f} Ko  (complet : {os.path.getsize(SRC) / 1e6:.1f} Mo)")

# --- versions : celles du NOTEBOOK pour tout ce qui touche au modèle et aux données
#     (l'artefact doit être relu avec les mêmes versions), celles d'ici pour le reste
nb_v = art.get("versions", {})
names = {"lightgbm": "lightgbm", "scikit-learn": "sklearn", "pandas": "pandas", "numpy": "numpy", "joblib": "joblib"}
pins = {}
for pkg, key in names.items():
    nb = nb_v.get(key)
    if nb is None:
        print(f"ATTENTION : version de {pkg} absente de l'artefact (relancer la cellule d'export du notebook) ; "
              f"version locale utilisée : {local(pkg)}")
    pins[pkg] = nb or local(pkg)
    if nb and local(pkg) != nb:
        print(f"ATTENTION {pkg} : notebook {nb} != local {local(pkg)}. Pour tester ici à l'identique : pip install {pkg}=={nb}")


def write_req(path, extra):
    lines = [f"{p}=={v}" for p, v in {**pins, **{e: local(e) for e in extra}}.items() if v]
    missing = [e for e in extra if not local(e)]
    open(path, "w").write("\n".join(sorted(lines)) + "\n")
    print(f"{path} : {len(lines)} paquets" + (f" (non installés ici, non épinglés : {missing})" if missing else ""))


write_req("requirements.txt", ["streamlit", "plotly"])                       # app (Streamlit Cloud)
write_req("requirements-api.txt", ["fastapi", "uvicorn", "pydantic"])        # API (Render)

pyv = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
open(".python-version", "w").write(pyv + "\n")
if os.path.exists("render.yaml"):
    y = open("render.yaml").read()
    open("render.yaml", "w").write(re.sub(r'(key: PYTHON_VERSION\s*\n\s*value: )\S+', rf"\g<1>{pyv}", y))
print("Python :", pyv)

# --- contrôle sur l'artefact allégé
fc.selfcheck(API_OUT)
print("\nPrêt à pousser : git add -A && git commit -m \"Mise à jour du modèle\" && git push")