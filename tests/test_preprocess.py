from ofensivo.preprocess import limpiar


def test_leet_se_revierte():
    assert limpiar("Eres un p3nd3j0") == "eres un pendejo"
    assert limpiar("idio7a") == "idiota"


def test_censura_con_simbolos_usa_lexico():
    assert limpiar("qué p*ta madre") == "que puta madre"
    assert limpiar("ch#ngada") == "chingada"


def test_conserva_negacion_y_enie():
    assert limpiar("No eres NIÑO") == "no eres niño"


def test_emoji_ofensivo_se_convierte_no_se_borra():
    assert "emoji_payaso" in limpiar("qué tonto 🤡")
    assert limpiar("hola 😀") == "hola"


def test_urls_menciones_y_numeros():
    assert limpiar("@juan mira https://x.com/a 5567854323 hola") == "mira hola"


def test_abreviaturas_y_alargamientos():
    assert limpiar("ptm q hueva holaaaa") == "puta madre que hueva hola"
