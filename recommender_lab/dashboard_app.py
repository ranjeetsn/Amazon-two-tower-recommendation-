import os

import numpy as np
import streamlit as st

from recommender_lab.storage import (
    feedback_count,
    get_dashboard_frames,
    init_db,
    last_trained_feedback_count,
    new_feedback_count,
)


LABEL_TO_RELEVANCE = {"I": 0, "C": 1, "S": 2, "E": 3}


def dcg_at_k(relevances, k=10):
    relevances = np.asarray(relevances, dtype=float)[:k]
    gains = (2.0 ** relevances) - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def model_ndcg(feedback, recommendations, k=10):
    if feedback.empty or recommendations.empty:
        return None

    labels = feedback.copy()
    labels["_human_priority"] = (labels["source"] == "human").astype(int)
    labels = labels.sort_values("_human_priority", ascending=False)
    labels = labels.drop_duplicates(["request_id", "product_id"], keep="first")

    joined = recommendations.merge(
        labels[["request_id", "product_id", "label", "source"]],
        on=["request_id", "product_id"],
        how="inner",
    )
    if joined.empty:
        return None

    output = []
    for (model_name, source), model_rows in joined.groupby(
        ["model_name", "source"], sort=False
    ):
        query_scores = []
        for _, query_rows in model_rows.groupby("request_id", sort=False):
            ordered = query_rows.sort_values("rank", ascending=True)
            relevance = [
                LABEL_TO_RELEVANCE.get(str(label).upper(), 0)
                for label in ordered["label"]
            ]
            ideal = sorted(relevance, reverse=True)
            ideal_dcg = dcg_at_k(ideal, k)
            query_scores.append(
                dcg_at_k(relevance, k) / ideal_dcg if ideal_dcg > 0 else 0.0
            )
        output.append(
            {
                "model": model_name,
                "feedback_source": source,
                "queries_labeled": len(query_scores),
                "ndcg_at_10": float(np.mean(query_scores)),
            }
        )
    return output


st.set_page_config(page_title="Recommender Results Dashboard", layout="wide")
init_db()

st.title("Recommendation results and feedback")
st.caption(
    "LLM-generated labels are synthetic relevance estimates. Review them separately "
    "from any human or eventual customer feedback."
)

if st.button("Refresh dashboard"):
    st.rerun()

searches, recommendations, feedback, training_runs = get_dashboard_frames()
new_count = new_feedback_count()
threshold = int(os.getenv("RETRAIN_THRESHOLD", "1000"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Queries", len(searches))
col2.metric("Labeled query-product pairs", feedback_count())
col3.metric("New labels since training", f"{new_count} / {threshold}")
col4.metric("Training status", training_runs.iloc[0]["status"] if not training_runs.empty else "not run")

if new_count >= threshold:
    st.warning("The retraining worker has enough new labels to start a training run.")

st.subheader("Ranking quality by approach")
metrics = model_ndcg(feedback, recommendations)
if metrics:
    st.dataframe(metrics, use_container_width=True, hide_index=True)
else:
    st.info("No labeled recommendations yet. Run a query and simulate or record feedback.")

st.subheader("Feedback label counts")
if not feedback.empty:
    st.bar_chart(feedback.groupby(["source", "label"]).size().unstack(fill_value=0))
    st.dataframe(
        feedback[["created_at", "request_id", "product_id", "product_title", "label", "source", "rationale"]].head(100),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No feedback has been recorded.")

st.subheader("Recent training runs")
if not training_runs.empty:
    st.dataframe(training_runs, use_container_width=True, hide_index=True)
else:
    st.write("No retraining runs have been recorded.")

with st.expander("Recent searches"):
    st.dataframe(searches.head(100), use_container_width=True, hide_index=True)
