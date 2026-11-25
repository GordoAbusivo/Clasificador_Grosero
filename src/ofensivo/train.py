"""Entrenamiento y evaluación.

Ejemplos:
    python -m ofensivo.train --modelo mlp
    python -m ofensivo.train --modelo cnn_char --augment 2
    python -m ofensivo.train --modelo mlp --tarea categoria
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .augment import aumentar
from .evaluate import errores, evaluar, graficar_historial, predecir_clases, umbral_por_recall
from .features import crear_vectorizador
from .preprocess import limpiar_corpus

MODELOS = ("logreg", "mlp", "cnn", "cnn_char")


@dataclass
class Config:
    datos: str = "data/seed_mx.csv"
    modelo: str = "mlp"            # logreg | mlp | cnn | cnn_char
    tarea: str = "binaria"         # binaria | categoria
    ocultas: tuple = (32,)         # () = regresión logística en Keras
    activacion: str = "relu"
    dropout: float = 0.5
    l2: float = 1e-3
    l1_entrada: float = 0.0
    epocas: int = 150
    batch: int = 32
    augment: int = 0               # copias ruidosas por texto (solo train)
    recall_min: float = 0.85
    semilla: int = 42
    nombre: str | None = None
    test_externo: str | None = None
    salida_modelos: str = "models"
    salida_reportes: str = "reports"


def _fijar_semillas(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(s)
    except ImportError:
        pass


def cargar_datos(ruta: str, tarea: str):
    df = pd.read_csv(ruta).dropna(subset=["texto", "etiqueta"])
    df = df.drop_duplicates(subset="texto").reset_index(drop=True)  # evita fuga por duplicados
    if tarea == "categoria":
        if "categoria" not in df:
            raise SystemExit("La tarea 'categoria' requiere una columna 'categoria' en el CSV.")
        etiquetas = sorted(df["categoria"].unique())
        y = df["categoria"].map({e: i for i, e in enumerate(etiquetas)}).to_numpy()
    else:
        etiquetas = ["no_ofensivo", "ofensivo"]
        y = df["etiqueta"].astype(int).to_numpy()
    return df["texto"].astype(str).to_numpy(), y, etiquetas


def dividir(y, semilla: int):
    """70 % train / 15 % validación / 15 % prueba, estratificado si es posible."""
    idx = np.arange(len(y))
    try:
        tr, resto = train_test_split(idx, test_size=0.30, random_state=semilla, stratify=y)
        va, te = train_test_split(resto, test_size=0.50, random_state=semilla, stratify=y[resto])
    except ValueError:
        tr, resto = train_test_split(idx, test_size=0.30, random_state=semilla)
        va, te = train_test_split(resto, test_size=0.50, random_state=semilla)
    return tr, va, te


def _probs_keras(modelo, X, cnn: bool, batch: int):
    if cnn:
        import tensorflow as tf

        p = modelo.predict(tf.data.Dataset.from_tensor_slices(list(X)).batch(batch), verbose=0)
    else:
        p = modelo.predict(X, verbose=0)
    return p.ravel() if p.shape[-1] == 1 else p


def entrenar(cfg: Config) -> dict:
    _fijar_semillas(cfg.semilla)
    nombre = cfg.nombre or f"{cfg.modelo}_{cfg.tarea}"
    dir_modelo = Path(cfg.salida_modelos) / nombre
    dir_rep = Path(cfg.salida_reportes) / nombre
    dir_modelo.mkdir(parents=True, exist_ok=True)
    dir_rep.mkdir(parents=True, exist_ok=True)

    X_raw, y, etiquetas = cargar_datos(cfg.datos, cfg.tarea)
    n_clases = len(etiquetas)
    tr, va, te = dividir(y, cfg.semilla)

    X_tr_raw, y_tr = X_raw[tr], y[tr]
    if cfg.augment:
        X_tr_raw, y_tr = aumentar(X_tr_raw, y_tr, cfg.augment, cfg.semilla)  # después de dividir
    X_tr, X_va, X_te = limpiar_corpus(X_tr_raw), limpiar_corpus(X_raw[va]), limpiar_corpus(X_raw[te])
    y_va, y_te = y[va], y[te]

    meta = {"config": asdict(cfg), "etiquetas": etiquetas, "umbral": 0.5, "nombre": nombre}
    es_cnn = cfg.modelo in ("cnn", "cnn_char")

    if cfg.modelo == "logreg":
        vect = crear_vectorizador()
        A_tr = vect.fit_transform(X_tr)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(A_tr, y_tr)
        prob = lambda X: (clf.predict_proba(vect.transform(X))[:, 1] if n_clases == 2
                          else clf.predict_proba(vect.transform(X)))
        joblib.dump({"vectorizador": vect, "clasificador": clf}, dir_modelo / "modelo.joblib")
    else:
        from tensorflow import keras

        from .models import construir_cnn, construir_mlp

        pesos = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)))
        parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        if es_cnn:
            import tensorflow as tf

            modelo = construir_cnn(X_tr, n_clases, nivel="caracter" if cfg.modelo == "cnn_char" else "palabra",
                                   dropout=cfg.dropout, l2=cfg.l2)
            ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr)).batch(cfg.batch)
            ds_va = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(cfg.batch)
            hist = modelo.fit(ds_tr, validation_data=ds_va, epochs=cfg.epocas, callbacks=[parada],
                              class_weight=pesos, verbose=0)
        else:
            vect = crear_vectorizador()
            A_tr = vect.fit_transform(X_tr).toarray()
            A_va = vect.transform(X_va).toarray()
            modelo = construir_mlp(A_tr.shape[1], n_clases, tuple(cfg.ocultas), cfg.activacion,
                                   cfg.dropout, cfg.l2, cfg.l1_entrada)
            hist = modelo.fit(A_tr, y_tr, validation_data=(A_va, y_va), epochs=cfg.epocas,
                              batch_size=cfg.batch, callbacks=[parada], class_weight=pesos, verbose=0)
            joblib.dump(vect, dir_modelo / "vectorizador.joblib")
        graficar_historial(hist.history, dir_rep)
        modelo.save(dir_modelo / "modelo.keras")
        if es_cnn:
            prob = lambda X: _probs_keras(modelo, X, True, cfg.batch)
        else:
            prob = lambda X: _probs_keras(modelo, vect.transform(X).toarray(), False, cfg.batch)

    # Umbral elegido en VALIDACIÓN (nunca en test) para priorizar recall del ofensivo.
    if n_clases == 2:
        meta["umbral"] = umbral_por_recall(y_va, prob(X_va), cfg.recall_min)
    (dir_modelo / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    p_te = prob(X_te)
    metricas = evaluar(y_te, p_te, etiquetas, dir_rep, meta["umbral"], titulo=f"({nombre})")
    errores(X_te, y_te, predecir_clases(p_te, meta["umbral"]), etiquetas, dir_rep)

    if cfg.test_externo and n_clases == 2:
        ext = pd.read_csv(cfg.test_externo).dropna(subset=["texto", "etiqueta"])
        p_ext = prob(limpiar_corpus(ext["texto"].astype(str)))
        metricas["externo"] = evaluar(ext["etiqueta"].astype(int), p_ext, etiquetas,
                                      dir_rep / "externo", meta["umbral"], titulo="(externo)")
    metricas["nombre"] = nombre
    metricas["n_train"], metricas["n_test"] = len(X_tr), len(X_te)
    return metricas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default=Config.datos)
    p.add_argument("--modelo", choices=MODELOS, default="mlp")
    p.add_argument("--tarea", choices=("binaria", "categoria"), default="binaria")
    p.add_argument("--ocultas", type=int, nargs="*", default=[32], help="neuronas por capa; vacío = regresión logística")
    p.add_argument("--activacion", default="relu", choices=("relu", "tanh", "sigmoid", "linear"))
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--l1-entrada", type=float, default=0.0)
    p.add_argument("--epocas", type=int, default=150)
    p.add_argument("--augment", type=int, default=0)
    p.add_argument("--recall-min", type=float, default=0.85)
    p.add_argument("--test-externo", default=None, help="CSV (texto,etiqueta) escrito a mano o real, NO sintético")
    p.add_argument("--nombre", default=None)
    a = p.parse_args()
    cfg = Config(datos=a.datos, modelo=a.modelo, tarea=a.tarea, ocultas=tuple(a.ocultas),
                 activacion=a.activacion, dropout=a.dropout, l2=a.l2, l1_entrada=a.l1_entrada,
                 epocas=a.epocas, augment=a.augment, recall_min=a.recall_min,
                 test_externo=a.test_externo, nombre=a.nombre)
    m = entrenar(cfg)
    print(json.dumps({k: v for k, v in m.items() if k != "matriz_confusion"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
