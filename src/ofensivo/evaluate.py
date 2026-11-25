"""Métricas y gráficas de evaluación."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


def umbral_por_recall(y_true, probs, recall_min: float = 0.85) -> float:
    """Umbral más alto que aún logra `recall_min` en la clase ofensiva (binario).

    En moderación pesa más NO dejar pasar un ataque (falso negativo) que una falsa alarma.
    """
    prec, rec, umbrales = precision_recall_curve(y_true, probs)
    validos = [u for u, r in zip(umbrales, rec[:-1]) if r >= recall_min]
    return float(max(validos)) if validos else 0.5


def predecir_clases(probs: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    probs = np.asarray(probs)
    return (probs >= umbral).astype(int) if probs.ndim == 1 else probs.argmax(axis=1)


def evaluar(y_true, probs, etiquetas: list[str], carpeta: Path, umbral: float = 0.5, titulo: str = "") -> dict:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = predecir_clases(probs, umbral)
    ids = list(range(len(etiquetas)))

    reporte = classification_report(y_true, y_pred, labels=ids, target_names=etiquetas,
                                    output_dict=True, zero_division=0)
    (carpeta / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=ids, target_names=etiquetas, zero_division=0),
        encoding="utf-8",
    )
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    fig, ax = plt.subplots(figsize=(5 + 0.4 * len(ids), 4 + 0.3 * len(ids)))
    ConfusionMatrixDisplay(cm, display_labels=etiquetas).plot(ax=ax, cmap="Blues", colorbar=False,
                                                              xticks_rotation=45)
    ax.set_title(f"Matriz de confusión {titulo}".strip())
    fig.tight_layout()
    fig.savefig(carpeta / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    metricas = {
        "accuracy": float(reporte["accuracy"]),
        "f1_macro": float(reporte["macro avg"]["f1-score"]),
        "umbral": float(umbral),
        "matriz_confusion": cm.tolist(),
    }
    if np.asarray(probs).ndim == 1 and len(etiquetas) == 2:
        pos = etiquetas[1]
        metricas.update(
            precision_ofensivo=float(reporte[pos]["precision"]),
            recall_ofensivo=float(reporte[pos]["recall"]),
            f1_ofensivo=float(reporte[pos]["f1-score"]),
            pr_auc=float(average_precision_score(y_true, probs)),
        )
        prec, rec, _ = precision_recall_curve(y_true, probs)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(rec, prec)
        ax.set(xlabel="Recall", ylabel="Precisión", title=f"Curva PR {titulo}".strip())
        fig.tight_layout()
        fig.savefig(carpeta / "curva_pr.png", dpi=150)
        plt.close(fig)
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    return metricas


def graficar_historial(historial: dict, carpeta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, clave, nombre in zip(ejes, ["loss", "accuracy"], ["Pérdida", "Accuracy"]):
        ax.plot(historial.get(clave, []), label="train")
        ax.plot(historial.get(f"val_{clave}", []), label="validación")
        ax.set(title=nombre, xlabel="época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(carpeta) / "historial.png", dpi=150)
    plt.close(fig)


def errores(textos, y_true, y_pred, etiquetas, carpeta: Path, max_n: int = 50) -> None:
    """Guarda los textos mal clasificados: es lo más útil para mejorar el dataset."""
    filas = ["texto\treal\tpredicho"]
    for t, r, p in zip(textos, y_true, y_pred):
        if r != p:
            filas.append(f"{t}\t{etiquetas[r]}\t{etiquetas[p]}")
    (Path(carpeta) / "errores.tsv").write_text("\n".join(filas[: max_n + 1]), encoding="utf-8")
