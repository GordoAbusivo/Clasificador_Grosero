"""Inferencia con un modelo entrenado.

    python -m ofensivo.predict --modelo models/mlp_binaria "Eres un inútil" "Qué chingón quedó"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .preprocess import limpiar_corpus


class Clasificador:
    def __init__(self, carpeta: str | Path):
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
        self.etiquetas = self.meta["etiquetas"]
        self.umbral = self.meta["umbral"]
        modelo = self.meta["config"]["modelo"]
        self._cnn = modelo in ("cnn", "cnn_char")
        if modelo == "logreg":
            d = joblib.load(carpeta / "modelo.joblib")
            self._vect, self._clf, self._keras = d["vectorizador"], d["clasificador"], None
        else:
            from tensorflow import keras

            self._keras = keras.models.load_model(carpeta / "modelo.keras")
            self._vect = None if self._cnn else joblib.load(carpeta / "vectorizador.joblib")

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        limpios = limpiar_corpus(textos)
        if self._keras is None:
            p = self._clf.predict_proba(self._vect.transform(limpios))
            return p[:, 1] if p.shape[1] == 2 else p
        if self._cnn:
            import tensorflow as tf

            p = self._keras.predict(tf.constant(limpios), verbose=0)
        else:
            p = self._keras.predict(self._vect.transform(limpios).toarray(), verbose=0)
        return p.ravel() if p.shape[-1] == 1 else p

    def predecir(self, textos: list[str]) -> list[dict]:
        p = self.probabilidades(textos)
        salida = []
        for t, pi in zip(textos, p):
            if np.ndim(pi) == 0:
                idx = int(pi >= self.umbral)
                salida.append({"texto": t, "etiqueta": self.etiquetas[idx], "probabilidad_ofensivo": float(pi)})
            else:
                idx = int(np.argmax(pi))
                salida.append({"texto": t, "etiqueta": self.etiquetas[idx], "probabilidad": float(pi[idx])})
        return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, help="carpeta en models/")
    ap.add_argument("textos", nargs="+")
    a = ap.parse_args()
    for r in Clasificador(a.modelo).predecir(a.textos):
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
