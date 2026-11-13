import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker

from recommender_lab.recommenders import (
    FEATURE_COLUMNS,
    RecommenderLab,
    load_products,
    read_table,
    tokenize,
)


LABEL_TO_RELEVANCE = {"I": 0, "C": 1, "S": 2, "E": 3}
LABEL_TO_BINARY = {"I": 0.0, "E": 1.0}


def _merge_product_fields(pairs, products):
    pairs = pairs.copy()
    pairs["product_id"] = pairs["product_id"].astype(str)
    catalog = products.copy()
    catalog["product_id"] = catalog["product_id"].astype(str)
    keep = ["product_id", "product_title"]
    if "product_brand" in catalog.columns:
        keep.append("product_brand")
    catalog = catalog[keep].drop_duplicates("product_id")

    merged = pairs.merge(catalog, on="product_id", how="left", suffixes=("", "_catalog"))
    if "product_title" not in merged.columns:
        merged["product_title"] = merged["product_title_catalog"]
    else:
        merged["product_title"] = merged["product_title"].fillna(
            merged.get("product_title_catalog")
        )
    if "product_brand_catalog" in merged.columns:
        if "product_brand" not in merged.columns:
            merged["product_brand"] = merged["product_brand_catalog"]
        else:
            merged["product_brand"] = merged["product_brand"].fillna(
                merged["product_brand_catalog"]
            )
    merged["product_title"] = merged["product_title"].fillna("").astype(str)
    if "product_brand" not in merged.columns:
        merged["product_brand"] = ""
    return merged


def _load_training_pairs(examples_path, products, feedback_df=None):
    examples = read_table(examples_path)
    if "query_id" not in examples.columns:
        examples["query_id"] = examples["query"].astype(str)

    # If the source file has an official split, never train on its held-out rows.
    if "split" in examples.columns:
        is_train = examples["split"].astype(str).str.lower().isin({"train", "training"})
        if is_train.any():
            examples = examples[is_train].copy()

    examples["esci_label"] = examples["esci_label"].astype(str).str.upper()
    examples["query_id"] = "esci:" + examples["query_id"].astype(str)
    examples = _merge_product_fields(examples, products)

    if feedback_df is not None and not feedback_df.empty:
        feedback = feedback_df.copy()
        feedback["query_id"] = "live:" + feedback["query_id"].astype(str)
        feedback["esci_label"] = feedback["esci_label"].astype(str).str.upper()
        feedback = _merge_product_fields(feedback, products)
        # Keep stored titles for any interaction product absent from the catalog snapshot.
        if "product_title_x" in feedback.columns and "product_title_y" in feedback.columns:
            feedback["product_title"] = (
                feedback["product_title_x"].fillna(feedback["product_title_y"]).fillna("")
            )
        examples = pd.concat([examples, feedback], ignore_index=True, sort=False)

    examples = examples[
        examples["esci_label"].isin(LABEL_TO_RELEVANCE)
        & examples["query"].notna()
        & examples["product_id"].notna()
    ].copy()
    examples["query"] = examples["query"].astype(str)
    examples["product_id"] = examples["product_id"].astype(str)
    examples["product_title"] = examples["product_title"].fillna("").astype(str)
    examples["product_brand"] = examples["product_brand"].fillna("").astype(str)
    return examples.reset_index(drop=True)


def _group_split(frame, test_size=0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, valid_idx = next(
        splitter.split(frame, groups=frame["query_id"].astype(str))
    )
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[valid_idx].reset_index(drop=True),
    )


def _sort_for_ranking(frame, features, labels):
    data = frame.assign(_label=labels.to_numpy())
    data = data.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    X = features.loc[data.index] if features.index.equals(data.index) else None
    # Rebuild features in exactly the same sorted row order.
    X = data[FEATURE_COLUMNS].copy() if set(FEATURE_COLUMNS).issubset(data.columns) else X
    group_sizes = data.groupby("query_id", sort=False).size().tolist()
    return data, group_sizes


def train_from_files(examples_path, products_path, feedback_df=None, artifact_root=None):
    products = load_products(products_path)
    pairs = _load_training_pairs(examples_path, products, feedback_df)
    if pairs["query_id"].nunique() < 2:
        raise ValueError("Need at least two distinct query groups to train and validate.")

    artifact_root = Path(
        artifact_root or os.getenv("RECOMMENDER_ARTIFACT_DIR", "recommender_lab/artifacts")
    )
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    version_dir = artifact_root / version
    version_dir.mkdir(parents=True, exist_ok=False)

    feature_builder = RecommenderLab(products, artifact_root=artifact_root, load_models=False)

    # XGBoost learns to order the four graded ESCI labels within each query.
    xgb_train, xgb_valid = _group_split(pairs)
    x_train = feature_builder.build_pair_features(xgb_train)
    x_valid = feature_builder.build_pair_features(xgb_valid)
    y_train = xgb_train["esci_label"].map(LABEL_TO_RELEVANCE).astype(int)
    y_valid = xgb_valid["esci_label"].map(LABEL_TO_RELEVANCE).astype(int)

    train_frame = xgb_train.copy()
    train_frame[FEATURE_COLUMNS] = x_train
    train_frame["_label"] = y_train
    train_frame = train_frame.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    valid_frame = xgb_valid.copy()
    valid_frame[FEATURE_COLUMNS] = x_valid
    valid_frame["_label"] = y_valid
    valid_frame = valid_frame.sort_values("query_id", kind="mergesort").reset_index(drop=True)

    train_groups = train_frame.groupby("query_id", sort=False).size().tolist()
    valid_groups = valid_frame.groupby("query_id", sort=False).size().tolist()

    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    ranker.fit(
        train_frame[FEATURE_COLUMNS],
        train_frame["_label"],
        group=train_groups,
        eval_set=[(valid_frame[FEATURE_COLUMNS], valid_frame["_label"])],
        eval_group=[valid_groups],
        verbose=False,
    )
    ranker.save_model(str(version_dir / "xgb_ranker.json"))

    # The first two-tower run uses only clear E/I pairs; S and C are not forced
    # into either binary class. Limit negatives per query to reduce imbalance.
    binary = pairs[pairs["esci_label"].isin(LABEL_TO_BINARY)].copy()
    positives = binary[binary["esci_label"] == "E"]
    negatives = binary[binary["esci_label"] == "I"]
    sampled_negatives = []
    for query_id, positive_group in positives.groupby("query_id"):
        query_negatives = negatives[negatives["query_id"] == query_id]
        if not query_negatives.empty:
            sampled_negatives.append(
                query_negatives.sample(
                    n=min(len(query_negatives), max(1, len(positive_group) * 4)),
                    random_state=42,
                )
            )
    if not sampled_negatives:
        raise ValueError("Two-tower training needs both E positives and I negatives.")
    binary = pd.concat([positives, *sampled_negatives], ignore_index=True)
    binary["label"] = binary["esci_label"].map(LABEL_TO_BINARY).astype(np.float32)
    tower_train, tower_valid = _group_split(binary)

    vocab_counter = {}
    for text in pd.concat(
        [tower_train["query"], tower_train["product_title"]], ignore_index=True
    ):
        for token in tokenize(text):
            vocab_counter[token] = vocab_counter.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in sorted(vocab_counter.items(), key=lambda item: (-item[1], item[0]))[:49_998]:
        vocab[token] = len(vocab)

    def encode(text, max_length):
        ids = [vocab.get(token, 1) for token in tokenize(text)[:max_length]]
        return ids + [0] * (max_length - len(ids))

    def encode_frame(frame):
        q = np.asarray([encode(text, 32) for text in frame["query"]], dtype=np.int32)
        p = np.asarray(
            [encode(text, 64) for text in frame["product_title"]], dtype=np.int32
        )
        y = frame["label"].to_numpy(dtype=np.float32).reshape(-1, 1)
        return q, p, y

    train_q, train_p, train_y = encode_frame(tower_train)
    valid_q, valid_p, valid_y = encode_frame(tower_valid)

    def make_tower(vocab_size, max_length, name):
        inputs = tf.keras.Input(shape=(max_length,), dtype="int32", name=f"{name}_tokens")
        x = tf.keras.layers.Embedding(
            vocab_size, 64, mask_zero=True, name=f"{name}_embedding"
        )(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_mean_pool")(x)
        x = tf.keras.layers.Dense(64, activation="relu", name=f"{name}_hidden")(x)
        x = tf.keras.layers.Dense(32, name=f"{name}_projection")(x)
        outputs = tf.keras.layers.UnitNormalization(axis=1, name=f"{name}_unit_vector")(x)
        return tf.keras.Model(inputs, outputs, name=name)

    query_tower = make_tower(len(vocab), 32, "query_tower")
    product_tower = make_tower(len(vocab), 64, "product_tower")
    query_input = tf.keras.Input(shape=(32,), dtype="int32", name="query_input")
    product_input = tf.keras.Input(shape=(64,), dtype="int32", name="product_input")
    similarity = tf.keras.layers.Dot(axes=1, name="cosine_similarity")(
        [query_tower(query_input), product_tower(product_input)]
    )
    logits = tf.keras.layers.Rescaling(10.0, name="scaled_similarity")(similarity)
    two_tower = tf.keras.Model([query_input, product_input], logits)
    two_tower.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    two_tower.fit(
        [train_q, train_p],
        train_y,
        validation_data=([valid_q, valid_p], valid_y),
        batch_size=256,
        epochs=10,
        callbacks=[early_stop],
        verbose=2,
    )

    query_tower.save(version_dir / "query_tower.keras")
    product_tower.save(version_dir / "product_tower.keras")
    (version_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )

    catalog_ids = products["product_id"].astype(str).tolist()
    catalog_titles = products["product_title"].fillna("").astype(str).tolist()
    product_inputs = np.asarray(
        [encode(title, 64) for title in catalog_titles], dtype=np.int32
    )
    embeddings = product_tower.predict(product_inputs, batch_size=512, verbose=0)
    np.savez_compressed(
        version_dir / "product_embeddings.npz",
        product_ids=np.asarray(catalog_ids, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )

    # Publish the version only after all artifacts have been written.
    pointer_tmp = artifact_root / "current.json.tmp"
    pointer_tmp.write_text(json.dumps({"version": version}), encoding="utf-8")
    pointer_tmp.replace(artifact_root / "current.json")
    return version


def main():
    examples_path = os.getenv("ESCI_EXAMPLES_PATH", "").strip()
    products_path = os.getenv("ESCI_PRODUCTS_PATH", "").strip()
    if not examples_path or not products_path:
        raise RuntimeError("Set ESCI_EXAMPLES_PATH and ESCI_PRODUCTS_PATH first.")

    feedback = None
    try:
        from recommender_lab.storage import load_training_feedback
        feedback = load_training_feedback()
    except Exception:
        pass

    version = train_from_files(
        examples_path,
        products_path,
        feedback_df=feedback,
    )
    print(f"Published recommender artifacts: {version}")


if __name__ == "__main__":
    main()
