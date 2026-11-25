"""Arquitecturas. Todas comparten la misma "cabeza" de salida:

- 2 clases  -> 1 neurona SIGMOIDE + entropía cruzada binaria
               (sin capa oculta = regresión logística)
- >2 clases -> N neuronas SOFTMAX + entropía cruzada categórica dispersa
               (sin capa oculta = regresión logística multinomial)
Una salida LINEAL + MSE sería una regresión: solo tendría sentido si
etiquetaran una "gravedad" numérica (p. ej. 0-10) en lugar de clases.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def _compilar(modelo: keras.Model, n_clases: int, lr: float) -> keras.Model:
    if n_clases == 2:
        modelo.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
        )
    else:
        modelo.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    return modelo


def _salida(n_clases: int):
    if n_clases == 2:
        return layers.Dense(1, activation="sigmoid", name="salida")
    return layers.Dense(n_clases, activation="softmax", name="salida")


def construir_mlp(
    dim_entrada: int,
    n_clases: int,
    ocultas: tuple[int, ...] = (32,),
    activacion: str = "relu",
    dropout: float = 0.5,
    l2: float = 1e-3,
    l1_entrada: float = 0.0,
    lr: float = 1e-3,
) -> keras.Model:
    """MLP sobre TF-IDF. `ocultas=()` equivale a una regresión logística.

    l1_entrada > 0 aplica L1 a la primera capa: apaga pesos de palabras inútiles
    (selección de características) y permite inspeccionar qué términos pesan más.
    """
    modelo = keras.Sequential([layers.Input(shape=(dim_entrada,))], name="mlp_tfidf")
    for i, n in enumerate(ocultas):
        reg = regularizers.l1_l2(l1=l1_entrada if i == 0 else 0.0, l2=l2)
        modelo.add(layers.Dense(n, activation=activacion, kernel_regularizer=reg))
        modelo.add(layers.Dropout(dropout))
    if not ocultas:
        modelo.add(layers.Dense(1 if n_clases == 2 else n_clases,
                                activation="sigmoid" if n_clases == 2 else "softmax",
                                kernel_regularizer=regularizers.l1_l2(l1=l1_entrada, l2=l2),
                                name="salida"))
    else:
        modelo.add(_salida(n_clases))
    return _compilar(modelo, n_clases, lr)


def construir_cnn(
    textos_entrenamiento,
    n_clases: int,
    nivel: str = "palabra",
    vocab: int = 6000,
    largo: int | None = None,
    emb: int = 64,
    filtros: int = 64,
    kernel: int | None = None,
    dropout: float = 0.5,
    l2: float = 1e-3,
    lr: float = 1e-3,
) -> keras.Model:
    """CNN 1D. Entrada: cadenas de texto ya limpiadas.

    nivel="caracter" es la variante recomendada para jerga/faltas: no depende del vocabulario.
    La capa TextVectorization va dentro del modelo (se guarda con él).
    """
    caracter = nivel == "caracter"
    largo = largo or (200 if caracter else 40)
    kernel = kernel or (5 if caracter else 3)
    vect = layers.TextVectorization(
        max_tokens=vocab if not caracter else 100,
        output_mode="int",
        output_sequence_length=largo,
        split="character" if caracter else "whitespace",
        standardize=None,
    )
    vect.adapt(list(textos_entrenamiento))

    entrada = keras.Input(shape=(), dtype="string", name="texto")
    x = vect(entrada)
    x = layers.Embedding(vect.vocabulary_size(), emb)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filtros, kernel, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    salida = _salida(n_clases)(x)
    modelo = keras.Model(entrada, salida, name=f"cnn_{nivel}")
    return _compilar(modelo, n_clases, lr)
