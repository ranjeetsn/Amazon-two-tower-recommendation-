import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_COLUMNS = [
    "exact_coverage",
    "fuzzy_score",
    "char_tfidf_score",
    "word_tfidf_score",
    "bm25_score",
    "query_token_count",
    "title_token_count",
    "brand_in_query",
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_products(path):
    products = read_table(path)
    required = {"product_id", "product_title"}
    missing = required - set(products.columns)
    if missing:
        raise ValueError(f"Product file is missing columns: {sorted(missing)}")
    keep = ["product_id", "product_title"]
    if "product_brand" in products.columns:
        keep.append("product_brand")
    products = products[keep].dropna(subset=["product_id"]).copy()
    products["product_id"] = products["product_id"].astype(str)
    products["product_title"] = products["product_title"].fillna("").astype(str)
    return products.drop_duplicates("product_id").reset_index(drop=True)


class RecommenderLab:
    def __init__(self, products, artifact_root=None, load_models=True):
        products = products.copy()
        products["product_id"] = products["product_id"].astype(str)
        products["product_title"] = products["product_title"].fillna("").astype(str)
        if "product_brand" not in products.columns:
            products["product_brand"] = ""
        products["product_brand"] = products["product_brand"].fillna("").astype(str)
        self.products = products.drop_duplicates("product_id").reset_index(drop=True)
        self.titles = self.products["product_title"].tolist()
        self.product_ids = self.products["product_id"].tolist()
        self.title_tokens = [tokenize(title) for title in self.titles]
        self.title_counters = [Counter(tokens) for tokens in self.title_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.title_tokens]
        self.doc_frequency = Counter()
        for tokens in self.title_tokens:
            self.doc_frequency.update(set(tokens))
        self.average_doc_length = (
            float(np.mean(self.doc_lengths)) if self.doc_lengths else 1.0
        )
        if self.average_doc_length <= 0:
            self.average_doc_length = 1.0

        # Fit lexical indexes once per app process, using catalog text only.
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=200_000
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=200_000
        )
        if any(self.titles):
            self.char_matrix = self.char_vectorizer.fit_transform(self.titles)
            self.word_matrix = self.word_vectorizer.fit_transform(self.titles)
        else:
            self.char_matrix = None
            self.word_matrix = None

        self.artifact_root = Path(
            artifact_root or os.getenv("RECOMMENDER_ARTIFACT_DIR", "recommender_lab/artifacts")
        )
        self.model_version = "lexical-only"
        self.xgb_ranker = None
        self.query_tower = None
        self.tower_vocab = None
        self.tower_embeddings = None
        self.tower_embedding_index = {}
        if load_models:
            self._load_active_models()

    def _active_dir(self):
        pointer = self.artifact_root / "current.json"
        if not pointer.exists():
            return None
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        version = metadata.get("version", "")
        active_dir = self.artifact_root / version
        if not active_dir.is_dir():
            return None
        self.model_version = version
        return active_dir

    def _load_active_models(self):
        active_dir = self._active_dir()
        if active_dir is None:
            return
        xgb_path = active_dir / "xgb_ranker.json"
        if xgb_path.exists():
            try:
                from xgboost import XGBRanker
                self.xgb_ranker = XGBRanker()
                self.xgb_ranker.load_model(str(xgb_path))
            except Exception:
                self.xgb_ranker = None

        query_path = active_dir / "query_tower.keras"
        vocab_path = active_dir / "vocab.json"
        embedding_path = active_dir / "product_embeddings.npz"
        if query_path.exists() and vocab_path.exists() and embedding_path.exists():
            try:
                import tensorflow as tf
                self.query_tower = tf.keras.models.load_model(str(query_path))
                self.tower_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
                stored = np.load(embedding_path, allow_pickle=False)
                ids = stored["product_ids"].astype(str)
                vectors = stored["embeddings"].astype(np.float32)
                self.tower_embedding_index = {
                    product_id: idx for idx, product_id in enumerate(ids)
                }
                self.tower_embeddings = vectors
            except Exception:
                self.query_tower = None
                self.tower_vocab = None
                self.tower_embeddings = None
                self.tower_embedding_index = {}

    def _bm25_doc_score(self, query_tokens, doc_tokens, doc_counter=None):
        if not query_tokens or not doc_tokens:
            return 0.0
        if doc_counter is None:
            doc_counter = Counter(doc_tokens)
        total = len(self.titles)
        score = 0.0
        k1, b = 1.5, 0.75
        doc_length = len(doc_tokens)
        for term in set(query_tokens):
            term_frequency = doc_counter.get(term, 0)
            if term_frequency == 0:
                continue
            df = self.doc_frequency.get(term, 0)
            inverse_frequency = np.log(
                1.0 + (total - df + 0.5) / (df + 0.5)
            )
            denominator = term_frequency + k1 * (
                1.0 - b + b * doc_length / self.average_doc_length
            )
            score += inverse_frequency * (
                term_frequency * (k1 + 1.0) / denominator
            )
        return float(score)

    def build_pair_features(self, pairs):
        """Create numerical query-product features for XGBoost training."""
        pairs = pairs.copy()
        queries = pairs["query"].fillna("").astype(str).tolist()
        titles = pairs["product_title"].fillna("").astype(str).tolist()
        query_tokens = [tokenize(query) for query in queries]
        title_tokens = [tokenize(title) for title in titles]

        exact = [
            len(set(q) & set(t)) / len(set(q)) if q else 0.0
            for q, t in zip(query_tokens, title_tokens)
        ]
        fuzzy = [fuzz.WRatio(q, t) / 100.0 for q, t in zip(queries, titles)]

        if self.char_matrix is not None:
            q_char = self.char_vectorizer.transform(queries)
            t_char = self.char_vectorizer.transform(titles)
            char_score = np.asarray(q_char.multiply(t_char).sum(axis=1)).ravel()
            q_word = self.word_vectorizer.transform(queries)
            t_word = self.word_vectorizer.transform(titles)
            word_score = np.asarray(q_word.multiply(t_word).sum(axis=1)).ravel()
        else:
            char_score = np.zeros(len(pairs))
            word_score = np.zeros(len(pairs))

        bm25 = [
            self._bm25_doc_score(q, t)
            for q, t in zip(query_tokens, title_tokens)
        ]
        brands = pairs.get("product_brand", pd.Series([""] * len(pairs))).fillna("").astype(str)
        brand_match = [
            float(bool(brand.strip()) and brand.strip().lower() in query.lower())
            for query, brand in zip(queries, brands)
        ]

        return pd.DataFrame(
            {
                "exact_coverage": exact,
                "fuzzy_score": fuzzy,
                "char_tfidf_score": char_score,
                "word_tfidf_score": word_score,
                "bm25_score": bm25,
                "query_token_count": [len(set(q)) for q in query_tokens],
                "title_token_count": [len(t) for t in title_tokens],
                "brand_in_query": brand_match,
            },
            index=pairs.index,
        )[FEATURE_COLUMNS]

    def _query_features(self, query):
        query = str(query)
        q_tokens = tokenize(query)
        query_set = set(q_tokens)
        exact = [
            len(query_set & set(tokens)) / len(query_set) if query_set else 0.0
            for tokens in self.title_tokens
        ]
        fuzzy = [fuzz.WRatio(query, title) / 100.0 for title in self.titles]

        if self.char_matrix is not None:
            q_char = self.char_vectorizer.transform([query])
            char = np.asarray((self.char_matrix @ q_char.T).toarray()).ravel()
            q_word = self.word_vectorizer.transform([query])
            word = np.asarray((self.word_matrix @ q_word.T).toarray()).ravel()
        else:
            char = np.zeros(len(self.products))
            word = np.zeros(len(self.products))

        bm25 = [
            self._bm25_doc_score(q_tokens, tokens, counter)
            for tokens, counter in zip(self.title_tokens, self.title_counters)
        ]
        brands = self.products["product_brand"].tolist()
        brand_match = [
            float(bool(brand.strip()) and brand.strip().lower() in query.lower())
            for brand in brands
        ]

        return pd.DataFrame(
            {
                "exact_coverage": exact,
                "fuzzy_score": fuzzy,
                "char_tfidf_score": char,
                "word_tfidf_score": word,
                "bm25_score": bm25,
                "query_token_count": [len(query_set)] * len(self.products),
                "title_token_count": self.doc_lengths,
                "brand_in_query": brand_match,
            }
        )[FEATURE_COLUMNS]

    def _tower_scores(self, query):
        if self.query_tower is None or self.tower_vocab is None or self.tower_embeddings is None:
            return None
        tokens = tokenize(query)[:32]
        token_ids = [self.tower_vocab.get(token, 1) for token in tokens]
        token_ids += [0] * (32 - len(token_ids))
        query_vector = self.query_tower.predict(
            np.asarray([token_ids], dtype=np.int32), verbose=0
        )[0]
        scores = np.full(len(self.products), -1.0, dtype=np.float32)
        for row, product_id in enumerate(self.product_ids):
            embedding_index = self.tower_embedding_index.get(product_id)
            if embedding_index is not None:
                scores[row] = float(
                    np.dot(query_vector, self.tower_embeddings[embedding_index])
                )
        return scores

    def recommend_all(self, query, k=10):
        features = self._query_features(query)
        scores = {
            "token_overlap": features["exact_coverage"].to_numpy(dtype=float),
            "fuzzy": features["fuzzy_score"].to_numpy(dtype=float),
            "char_tfidf": features["char_tfidf_score"].to_numpy(dtype=float),
            "word_tfidf": features["word_tfidf_score"].to_numpy(dtype=float),
            "bm25": features["bm25_score"].to_numpy(dtype=float),
        }

        if self.xgb_ranker is not None:
            scores["xgboost_ranker"] = self.xgb_ranker.predict(features)
        tower_scores = self._tower_scores(query)
        if tower_scores is not None:
            scores["two_tower"] = tower_scores

        results = {}
        for model_name, model_scores in scores.items():
            model_scores = np.nan_to_num(
                np.asarray(model_scores, dtype=float),
                nan=-np.inf,
                posinf=-np.inf,
                neginf=-np.inf,
            )
            top_indices = sorted(
                range(len(model_scores)),
                key=lambda idx: (-model_scores[idx], self.product_ids[idx]),
            )[:k]
            results[model_name] = [
                {
                    "product_id": self.product_ids[idx],
                    "product_title": self.titles[idx],
                    "score": float(model_scores[idx]),
                }
                for idx in top_indices
            ]
        return results
