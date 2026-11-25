"""Entrena varios modelos con el MISMO split y escribe reports/comparacion.md.

    python -m ofensivo.compare --augment 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .train import Config, entrenar

EXPERIMENTOS = {
    "logreg_sklearn": dict(modelo="logreg"),
    "logistica_keras (sin capa oculta)": dict(modelo="mlp", ocultas=()),
    "mlp_relu_32": dict(modelo="mlp", ocultas=(32,)),
    "mlp_tanh_32": dict(modelo="mlp", ocultas=(32,), activacion="tanh"),
    "mlp_relu_L1": dict(modelo="mlp", ocultas=(32,), l1_entrada=1e-4),
    "cnn_palabra": dict(modelo="cnn"),
    "cnn_caracter": dict(modelo="cnn_char"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="data/seed_mx.csv")
    p.add_argument("--augment", type=int, default=0)
    p.add_argument("--test-externo", default=None)
    a = p.parse_args()

    filas = ["| Modelo | Accuracy | F1 macro | Precisión (ofensivo) | Recall (ofensivo) | PR-AUC |",
             "|---|---|---|---|---|---|"]
    for etiqueta, kw in EXPERIMENTOS.items():
        nombre = "cmp_" + etiqueta.split()[0]
        m = entrenar(Config(datos=a.datos, augment=a.augment, test_externo=a.test_externo, nombre=nombre, **kw))
        filas.append(f"| {etiqueta} | {m['accuracy']:.3f} | {m['f1_macro']:.3f} | "
                     f"{m['precision_ofensivo']:.3f} | {m['recall_ofensivo']:.3f} | {m['pr_auc']:.3f} |")
        print(filas[-1])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/comparacion.md").write_text("\n".join(filas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
