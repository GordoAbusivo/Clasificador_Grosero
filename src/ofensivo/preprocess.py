"""Normalización de texto para español mexicano de redes sociales.

Decisiones de diseño (ver docs/recomendacion_modelos.md):
- NO se eliminan stopwords: "no", "tu", "eres" cambian el sentido de un insulto.
- Los emojis con carga ofensiva se CONVIERTEN en tokens (🤡 -> emoji_payaso), no se borran.
- Groserías censuradas con símbolos (p*ta, ch#ngada) se decodifican con un léxico.
- Leetspeak (p3nd3j0, idio7a) se revierte a letras.
- Abreviaturas ofensivas muy comunes (ptm, alv, hdp...) se expanden.
- Los acentos se eliminan (en redes casi nadie los usa) pero se conserva la ñ.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "lexicon_groserias_mx.txt"

# Emojis con señal para la tarea. El resto se elimina.
EMOJI_MAP = {
    "🤡": " emoji_payaso ",
    "🖕": " emoji_dedo ",
    "💩": " emoji_popo ",
    "🐷": " emoji_cerdo ",
    "🐖": " emoji_cerdo ",
    "🐽": " emoji_cerdo ",
    "🐀": " emoji_rata ",
    "🐒": " emoji_mono ",
    "🦍": " emoji_mono ",
    "🤮": " emoji_asco ",
    "🤢": " emoji_asco ",
    "🙄": " emoji_desprecio ",
    "😒": " emoji_desprecio ",
    "😂": " emoji_risa ",
    "🤣": " emoji_risa ",
    "😡": " emoji_enojo ",
    "🤬": " emoji_enojo ",
    "💀": " emoji_calavera ",
    "🔪": " emoji_cuchillo ",
    "🔫": " emoji_pistola ",
    "❤": " emoji_corazon ",
    "😍": " emoji_corazon ",
    "🥰": " emoji_corazon ",
}

# Abreviaturas / jerga que conviene normalizar (tras limpiar).
ABREVIATURAS = {
    "q": "que", "k": "que", "ke": "que", "d": "de",
    "pq": "porque", "xq": "porque", "porq": "porque",
    "tmb": "tambien", "tb": "tambien", "xfa": "por favor", "pls": "por favor",
    "ptm": "puta madre", "lpm": "la puta madre", "alv": "a la verga",
    "ctm": "chinga tu madre", "csm": "chingue su madre", "hdp": "hijo de puta",
    "nmms": "no mames", "nmm": "no mames",
    "we": "wey", "guey": "wey", "gey": "wey", "wuey": "wey",
}

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
)
_WILDCARDS = set("*#%&")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|mx|org|net)\b\S*)", re.I)
_MENCION_RE = re.compile(r"(?<!\w)@\w+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile("[\U00010000-\U0010FFFF☀-➿︀-️‍]")
# palabra, posiblemente con símbolos *dentro* (p*ta, p@ta, idio7a)
_TOKEN_RE = re.compile(r"\w+(?:[@$!*#%&]+\w+)*")


def quitar_acentos(texto: str) -> str:
    """Quita acentos y diéresis pero conserva la ñ."""
    salida: list[str] = []
    for ch in unicodedata.normalize("NFD", texto):
        if unicodedata.category(ch) == "Mn":
            if ch == "̃" and salida and salida[-1] == "n":
                salida[-1] = "ñ"
            continue
        salida.append(ch)
    return "".join(salida)


@lru_cache(maxsize=1)
def _lexico() -> tuple[str, ...]:
    if not LEXICON_PATH.exists():
        return ()
    palabras = []
    for linea in LEXICON_PATH.read_text(encoding="utf-8").splitlines():
        linea = linea.strip().lower()
        if linea and not linea.startswith("#"):
            palabras.append(quitar_acentos(linea))
    return tuple(palabras)


@lru_cache(maxsize=50_000)
def _decodificar_token(token: str) -> str:
    t = token.lower()
    if t.isdigit():
        return t
    letras = sum(c.isalpha() for c in t)
    tiene_simbolo = any(c in "@$!*#%&" for c in t)
    tiene_digito = any(c.isdigit() for c in t)
    if letras < 3 or not (tiene_simbolo or tiene_digito):
        return t
    t = t.translate(_LEET)
    if any(c in _WILDCARDS for c in t):
        patron = re.compile("".join("[a-zñ]" if c in _WILDCARDS else re.escape(c) for c in t))
        for palabra in _lexico():
            if patron.fullmatch(palabra):
                return palabra
        t = "".join(c for c in t if c not in _WILDCARDS)
    return t


def limpiar(texto: str) -> str:
    """Normaliza un texto crudo. Idempotente en lo esencial."""
    if not isinstance(texto, str):
        texto = "" if texto is None else str(texto)
    t = _HTML_RE.sub(" ", texto)
    t = _URL_RE.sub(" ", t)
    t = _MENCION_RE.sub(" ", t)
    for emoji, token in EMOJI_MAP.items():
        t = t.replace(emoji, token)
    t = _EMOJI_RE.sub(" ", t)
    t = t.lower()
    t = _TOKEN_RE.sub(lambda m: _decodificar_token(m.group(0)), t)
    t = quitar_acentos(t)
    t = re.sub(r"([a-zñ])\1{2,}", r"\1", t)          # puuuuta -> puta, noooo -> no
    t = re.sub(r"[^a-z0-9ñ_\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)                     # números sueltos (teléfonos, etc.)
    palabras = [ABREVIATURAS.get(p, p) for p in t.split()]
    return " ".join(palabras)


def limpiar_corpus(textos: Iterable[str]) -> list[str]:
    return [limpiar(t) for t in textos]
