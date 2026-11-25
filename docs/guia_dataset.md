# Guía para construir el dataset propio

El profesor pidió español de México y sugirió crear el dataset propio. La calidad de los datos pesa más que la arquitectura.

## 1. Definir "ofensivo" antes de etiquetar

Escriban una guía de una página y úsenla todos. Propuesta:

- **Ofensivo (1)**: texto dirigido a una persona o grupo para humillar, excluir, amenazar, acosar o discriminar (por clase, aspecto, origen, género…). Puede no tener ninguna grosería.
- **No ofensivo (0)**: todo lo demás, **incluidas groserías** sin víctima (hipérbole, desahogo, autocrítica, halago coloquial, conversación *sobre* el bullying).

Categorías (columna `categoria`): `groseria_dirigida`, `clasismo`, `racismo`, `misoginia_acoso`, `apariencia`, `exclusion`, `amenaza`, `sarcasmo_hostil`, `neutro`, `groseria_no_ofensiva`.

Etiqueten una muestra de 100 textos entre dos personas y calculen Cohen's κ (`sklearn.metrics.cohen_kappa_score`). Si κ < 0.6, la guía es ambigua. Ese número lo puedes poner en el reporte.

## 2. Fuentes (de más a menos recomendable)

| Fuente | Ventajas | Cuidado |
|---|---|---|
| Escritura propia del equipo | Español de ustedes, control total | Sesgo a "su" forma de hablar; varíen autores. |
| Generación con un LLM **+ revisión humana** | Rápido, escala | Ver abajo. |
| Comentarios públicos copiados a mano (anonimizados) | Realismo | Quitar usuarios, nombres y links; no subirlos al repo. |
| Scraping / APIs | Volumen | X exige API de pago, Reddit pide aprobación, Meta lo bloquea; además hay términos de uso y privacidad. Si lo hacen, guarden en `data/raw/` (ya ignorado por git). |
| Datasets públicos | Gratis | Casi todos son de España o en inglés; úsenlos solo como apoyo, citando la fuente. |

### Sobre las "groserías ñoñas" del LLM

El riesgo que menciona el profesor es real: un LLM suele producir insultos genéricos y pulcros que nadie escribe. Para mitigarlo:

1. Dale 15–20 ejemplos **reales** del equipo como *few-shot* y pide variedad de registro (mensaje de WhatsApp, comentario de TikTok, chat de videojuego).
2. Pide **pares mínimos**: la misma grosería en un caso ofensivo y en otro no.
3. Pide ofensas **sin groserías** (clasismo, sarcasmo) en proporción alta.
4. Revisen a mano y borren lo que "suene a robot". Descarten el 20–30 %.
5. **Nunca midan solo sobre datos sintéticos**: armen un conjunto de prueba aparte (≥150–200 textos) escrito o recolectado a mano y úsenlo con `--test-externo`.

## 3. Cómo evitar resultados inflados

- Deduplicar antes de dividir (el código lo hace).
- **Dividir primero, aumentar después**, solo en train (`--augment`).
- No repetir una misma frase cientos de veces para "reforzar" un patrón: aparece en train **y** en test y la métrica sube sin que el modelo aprenda nada.
- Evitar que plantillas casi idénticas (misma frase con otro nombre) queden en ambos lados.
- Equilibrio de clases razonable; si no, usar pesos de clase (el código ya los calcula).

## 4. Símbolos, emoticones y leetspeak

- `p*ta`, `ch#ngada` → se resuelven contra `data/lexicon_groserias_mx.txt` (amplíenlo).
- `p3nd3j0`, `idio7a` → reversión de leet.
- `🤡`, `🖕`, `💩`, `🐷` → tokens `emoji_*` (editar `EMOJI_MAP` en `preprocess.py`). Borrarlos pierde señal; un 🤡 tras una frase cambia el sentido.
- Un diccionario de emoticones con groserías sirve como **característica adicional**, no como reemplazo del clasificador.

## 5. Tamaño razonable

`data/seed_mx.csv` (100 textos) es solo una **semilla de ejemplo** para que el código corra. Con 100 ejemplos las métricas son ruido (el test tiene 15 filas). Apunten a ≥2 000 textos por clase para el binario y ≥150 por categoría si quieren clasificar categorías.
