"""Vectorización TF-IDF de palabras + caracteres.

Los n-gramas de caracteres hacen al modelo robusto a faltas de ortografía,
alargamientos y variantes ("pndejo", "cabronazo") que un vocabulario de
palabras exactas no ve.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def crear_vectorizador(max_palabras: int = 3000, max_chars: int = 3000) -> FeatureUnion:
    palabras = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_palabras,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",  # conserva tokens de 1-2 letras como "no", "tu"
    )
    caracteres = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=max_chars,
        sublinear_tf=True,
    )
    return FeatureUnion([("palabras", palabras), ("caracteres", caracteres)])
