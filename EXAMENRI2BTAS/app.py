from collections import Counter, defaultdict
from pathlib import Path
import re

from flask import Flask, jsonify, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stemmer = SnowballStemmer("english")
stopwords_ingles = set(stopwords.words("english"))
patron_tokens = re.compile(r"(?u)\b[a-z0-9][a-z0-9]+\b")


def preprocesar(texto):
    tokens = patron_tokens.findall(str(texto).lower())
    return [
        stemmer.stem(token)
        for token in tokens
        if token not in stopwords_ingles
    ]


corpus_archivos = list(CORPUS_DIR.glob("*.csv"))
if len(corpus_archivos) != 1:
    raise RuntimeError(
        f"Se esperaba un único CSV en {CORPUS_DIR}; "
        f"se encontraron {len(corpus_archivos)}."
    )

corpus = pd.read_csv(corpus_archivos[0], encoding="utf-8")
faltantes = {"_id", "title", "text"}.difference(corpus.columns)
if faltantes:
    raise RuntimeError(f"Faltan columnas del corpus: {sorted(faltantes)}")

corpus = corpus[["_id", "title", "text"]].copy()
corpus["_id"] = corpus["_id"].astype(str)
corpus["title"] = corpus["title"].fillna("").astype(str)
corpus["text"] = corpus["text"].fillna("").astype(str)

if corpus["_id"].duplicated().any():
    raise RuntimeError("El corpus contiene identificadores duplicados.")

document_ids = corpus["_id"].tolist()
titles = corpus["title"].tolist()
original_texts = corpus["text"].tolist()

search_texts = (
    corpus["title"].str.strip() + "\n" + corpus["text"].str.strip()
).str.strip()
token_lists = search_texts.map(preprocesar).tolist()
processed_texts = [" ".join(tokens) for tokens in token_lists]

document_lengths = [len(tokens) for tokens in token_lists]
longitud_media = sum(document_lengths) / len(document_lengths)

indice_invertido = defaultdict(list)
for doc_pos, tokens in enumerate(token_lists):
    for termino, frecuencia in Counter(tokens).items():
        indice_invertido[termino].append((doc_pos, frecuencia))

tfidf_vectorizer = TfidfVectorizer(
    lowercase=False,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False,
    norm="l2",
)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)


def preparar_resultados(puntuaciones, top_k=10):
    top_k = max(1, min(int(top_k), len(document_ids)))
    orden = np.argsort(
        -np.asarray(puntuaciones, dtype=float),
        kind="stable",
    )

    resultados = []
    ids_vistos = set()
    for doc_pos in orden:
        doc_id = document_ids[int(doc_pos)]
        if doc_id in ids_vistos:
            continue

        ids_vistos.add(doc_id)
        resultados.append({
            "rank": len(resultados) + 1,
            "document_id": doc_id,
            "score": float(puntuaciones[int(doc_pos)]),
            "title": titles[int(doc_pos)],
            "snippet": original_texts[int(doc_pos)][:300].strip(),
        })

        if len(resultados) == top_k:
            break
    return resultados


def buscar_tfidf(query, top_k=10):
    processed_query = " ".join(preprocesar(query))
    query_vector = tfidf_vectorizer.transform([processed_query])
    puntuaciones = cosine_similarity(query_vector, tfidf_matrix).ravel()
    return preparar_resultados(puntuaciones, top_k)


def buscar_bm25(query, top_k=10, k1=1.2, b=0.75):
    puntuaciones = np.zeros(len(document_ids), dtype=float)
    total_documentos = len(document_ids)

    for termino in set(preprocesar(query)):
        postings = indice_invertido.get(termino, [])
        if not postings:
            continue

        frecuencia_documental = len(postings)
        idf = np.log(
            1
            + (total_documentos - frecuencia_documental + 0.5)
            / (frecuencia_documental + 0.5)
        )

        for doc_pos, frecuencia in postings:
            denominador = frecuencia + k1 * (
                1 - b + b * document_lengths[doc_pos] / longitud_media
            )
            aporte = idf * (frecuencia * (k1 + 1)) / denominador
            puntuaciones[doc_pos] += aporte

    return preparar_resultados(puntuaciones, top_k)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "corpus": "NFCorpus",
        "documents": len(document_ids),
    })


@app.post("/search")
def search():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({
            "error": "El cuerpo debe ser un objeto JSON."
        }), 400

    query = body.get("query")
    model = body.get("model")
    top_k = body.get("top_k", 10)

    if not isinstance(query, str) or not query.strip():
        return jsonify({
            "error": "El campo query debe ser texto y no puede estar vacío."
        }), 400

    if model not in ("tfidf", "bm25"):
        return jsonify({
            "error": "El campo model debe ser 'tfidf' o 'bm25'."
        }), 400

    if isinstance(top_k, bool) or not isinstance(top_k, (int, float)):
        return jsonify({
            "error": "El campo top_k debe ser un entero positivo."
        }), 400

    top_k = int(top_k)
    if top_k < 1:
        return jsonify({
            "error": "El campo top_k debe ser un entero positivo."
        }), 400

    if model == "tfidf":
        results = buscar_tfidf(query.strip(), top_k)
    else:
        results = buscar_bm25(query.strip(), top_k)

    return jsonify({
        "model": model,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
