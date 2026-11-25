# Qué red usar (y por qué)

## Resumen

| Prioridad | Modelo | Para qué sirve en el proyecto |
|---|---|---|
| Base de referencia | Regresión logística (sklearn) sobre TF-IDF | Si tu red no la supera, la red no se justifica. |
| **Modelo principal (recomendado)** | **MLP sobre TF-IDF de palabras + caracteres**, salida sigmoide | Poca data, se entrena en segundos, fácil de regularizar y explicar. |
| Comparativo "visto en clase" | CNN 1D (palabra y **carácter**) | Muestra que sabes usar convoluciones; la de carácter aguanta faltas y leetspeak. |
| Opcional (fuera del alcance) | BETO / transformer en español | Es el estado del arte; solo mencionarlo en la discusión. |

Con menos de ~5 000 ejemplos lo normal es que **la regresión logística y el MLP queden casi empatados** y que la CNN solo gane si tus datos tienen mucha variación de escritura. Eso no es un fracaso: es un resultado que se reporta (`python -m ofensivo.compare`).

## Activaciones: qué equivale a qué

| Idea tuya | Equivalencia real | Cuándo usarla aquí |
|---|---|---|
| Sigmoide ≈ regresión logística | Una sola neurona sigmoide **sin capa oculta** + entropía cruzada binaria **es** regresión logística | **Capa de salida** de clasificación binaria (ofensivo / no). |
| Softmax | Regresión logística multinomial | Salida si clasificas por **categoría** (clasismo, amenaza, apariencia…). |
| Lineal ≈ regresión lineal | Neurona lineal + MSE **es** regresión lineal | Solo si etiquetas una **gravedad numérica** (0–10). Para clases, no. |
| ReLU | Oculta por defecto | **Capas ocultas.** Barata y no satura. |
| Tanh | Oculta centrada en 0 | Alternativa para comparar (`--activacion tanh`). Con TF-IDF casi no cambia. |
| Sigmoide en ocultas | — | Evitarla: gradientes que se desvanecen. |
| Lineal en ocultas | Colapsa a una regresión logística | No aporta nada: sin no linealidad, más capas no sirven. |

En el repo: `--ocultas` (vacío) = regresión logística en Keras; `--activacion tanh` para comparar.

## Arquitectura recomendada

```
texto crudo → limpiar() → TF-IDF (palabras 1-2g + caracteres 2-5g) → Dense(32, ReLU, L2) → Dropout(0.5) → Dense(1, sigmoide)
```

- **Caracteres 2–5**: cubren "pndejo", "cabronazooo", "ch1ngada" sin lista de sinónimos.
- **Sin quitar stopwords**: "no", "tú", "eres" cambian el sentido de un insulto.
- Una capa oculta pequeña basta. Si el *val_loss* sube pronto, achica (16) o sube Dropout, no agregues capas.

## Regularización (lo que ya viste, dónde aplica)

- **L2** en las capas densas (`--l2`): reduce pesos grandes. Siempre.
- **L1** en la primera capa (`--l1-entrada`): apaga palabras irrelevantes → selección de características e interpretación.
- **Dropout** (`--dropout`): 0.4–0.6 en esta escala de datos.
- **Early stopping** sobre `val_loss` con `restore_best_weights`.
- **Pesos de clase** si hay desbalance; **label smoothing** pequeño (0.05) por etiquetas ruidosas.

## Métricas

1. **Matriz de confusión** y **recall de la clase ofensiva**: un falso negativo es un ataque que pasó.
2. **Precisión del ofensivo**: falsos positivos = groserías inocentes censuradas ("qué chingón").
3. **F1** y **PR-AUC** (no dependen del umbral).
4. El **umbral** se fija en *validación* (`--recall-min`), nunca en test.
5. Revisa `reports/<modelo>/errores.tsv`: ahí está lo que realmente mejora el dataset.

## Qué problemas debe resolver el modelo (y cómo los cubre el repo)

| Reto del profesor | Cobertura |
|---|---|
| Groserías ≠ ofensa ("qué chingón", "estoy hasta la madre") | Clase `groseria_no_ofensiva` en el dataset: **negativos difíciles**. Es lo más importante del proyecto. |
| Insulto sin groserías (clasismo, sarcasmo, apariencia) | Categorías `clasismo`, `sarcasmo_hostil`, `apariencia`… con ejemplos sin una sola mala palabra. |
| Groserías con símbolos (`p*ta`, `ch#ngada`) | `preprocess.py` las decodifica con `data/lexicon_groserias_mx.txt`. |
| Emojis / emoticones | Se **convierten** a tokens (`🤡 → emoji_payaso`), no se borran. |
| Leetspeak y abreviaturas (`p3nd3j0`, `ptm`, `hdp`) | Reversión de leet + expansión de abreviaturas. |
