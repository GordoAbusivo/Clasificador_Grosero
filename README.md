# Clasificador_Grosero

Clasificador de texto **ofensivo / no ofensivo** (groserías dirigidas, clasismo, acoso, apariencia, exclusión, amenazas…) en **español de México**, con redes neuronales en Keras y una línea base de regresión logística.

> Proyecto de Machine Learning. El código y el dataset semilla son originales; no reutilizan material de otros autores.

## Qué incluye

- Preprocesamiento pensado para el español de redes: leetspeak, groserías censuradas con símbolos, emojis como señal, abreviaturas (`ptm`, `alv`, `hdp`), sin quitar stopwords.
- Modelos: regresión logística, **MLP sobre TF-IDF** (recomendado), CNN 1D de palabra y de **carácter**.
- Tarea binaria o por **categoría** (`--tarea categoria`).
- Evaluación: matriz de confusión, reporte por clase, curva PR, umbral por recall, lista de errores.
- Aumento de datos **solo en train** (sin fuga).
- Comparador automático de modelos.

La recomendación razonada de qué red usar está en [`docs/recomendacion_modelos.md`](docs/recomendacion_modelos.md) y cómo construir el dataset en [`docs/guia_dataset.md`](docs/guia_dataset.md).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                               # Windows PowerShell: $env:PYTHONPATH="src"
pytest -q
```

## Uso

```bash
# Entrenar el modelo recomendado
python -m ofensivo.train --modelo mlp

# Variantes
python -m ofensivo.train --modelo mlp --ocultas            # sin capa oculta = regresión logística
python -m ofensivo.train --modelo mlp --activacion tanh
python -m ofensivo.train --modelo mlp --l1-entrada 1e-4
python -m ofensivo.train --modelo cnn_char --augment 2
python -m ofensivo.train --modelo mlp --tarea categoria

# Evaluar con un conjunto externo escrito/recolectado a mano (recomendado)
python -m ofensivo.train --modelo mlp --test-externo data/test_externo.csv

# Comparar todos con el mismo split -> reports/comparacion.md
python -m ofensivo.compare --augment 1

# Predecir
python -m ofensivo.predict --modelo models/mlp_binaria "Eres un inútil" "Qué chingón quedó tu proyecto"
```

Formato del CSV: `texto,etiqueta[,categoria]` con `etiqueta` 0/1.

## Estructura

```
data/            seed_mx.csv (100 textos de ejemplo) y léxico de groserías
docs/            recomendación de modelos y guía del dataset
src/ofensivo/    preprocess, features, augment, models, train, evaluate, predict, compare
tests/           pruebas del preprocesamiento
reports/         se generan métricas y gráficas por modelo
models/          se guardan los modelos entrenados (ignorado por git)
```

## Advertencia importante

`data/seed_mx.csv` solo sirve para verificar que el código corre: con 100 textos el conjunto de prueba tiene 15 filas y las métricas **no significan nada**. Reemplázalo con tu dataset (ver la guía) y evalúa con un conjunto externo.

## Subir a GitHub

```bash
git init -b main
git add .
git commit -m "Proyecto inicial: clasificador de texto ofensivo en español de México"
git remote add origin https://github.com/GordoAbusivo/Clasificador_Grosero.git
```
