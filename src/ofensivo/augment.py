"""Aumento de datos. SE APLICA SOLO AL CONJUNTO DE ENTRENAMIENTO, después de dividir.

Aumentar antes de dividir mete copias casi idénticas en train y test (fuga de datos)
y las métricas salen infladas.
"""
from __future__ import annotations

import random
import re

import numpy as np

JERGA = {
    "wey": ["carnal", "compa", "bro", "vato"],
    "amigo": ["compa", "carnal", "bro"],
    "malo": ["manco", "tronco"],
    "tonto": ["menso", "wey"],
    "mucho": ["un chingo", "bastante"],
}
ABREV = {" que ": [" q ", " k ", " ke "], " porque ": [" pq ", " xq "], " también ": [" tmb "]}
VECINAS = {"b": "v", "v": "b", "s": "z", "z": "s", "c": "s", "y": "ll", "ll": "y"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "t": "7"}
ACENTOS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def sin_acentos(t: str, rng: random.Random) -> str:
    return t.translate(ACENTOS)


def jerga(t: str, rng: random.Random) -> str:
    for palabra, opciones in JERGA.items():
        t = re.sub(rf"\b{palabra}\b", lambda m: rng.choice(opciones), t, flags=re.I)
    return t


def abreviar(t: str, rng: random.Random) -> str:
    t = f" {t} "
    for largo, cortos in ABREV.items():
        t = t.replace(largo, rng.choice(cortos))
    return t.strip()


def alargar(t: str, rng: random.Random) -> str:
    palabras = t.split()
    if not palabras:
        return t
    i = rng.randrange(len(palabras))
    w = palabras[i]
    if w and w[-1].isalpha():
        palabras[i] = w + w[-1] * rng.randint(1, 3)
    return " ".join(palabras)


def confundir(t: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(t) if c.lower() in VECINAS]
    if not idx:
        return t
    i = rng.choice(idx)
    return t[:i] + VECINAS[t[i].lower()] + t[i + 1 :]


def leet(t: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(t) if c.lower() in LEET]
    if len(idx) < 2:
        return t
    for i in rng.sample(idx, k=min(2, len(idx))):
        t = t[:i] + LEET[t[i].lower()] + t[i + 1 :]
    return t


TRANSFORMACIONES = [sin_acentos, jerga, abreviar, alargar, confundir, leet]


def transformar(texto: str, rng: random.Random, p: float = 0.4) -> str:
    for f in TRANSFORMACIONES:
        if rng.random() < p:
            texto = f(texto, rng)
    return texto


def aumentar(textos, y, factor: int = 1, semilla: int = 42):
    """Devuelve (textos + copias ruidosas, etiquetas). `factor` = copias por texto."""
    rng = random.Random(semilla)
    textos = list(textos)
    y = np.asarray(y)
    nuevos, ny = list(textos), list(y)
    for _ in range(factor):
        for t, lab in zip(textos, y):
            nuevos.append(transformar(t, rng))
            ny.append(lab)
    return nuevos, np.asarray(ny)
