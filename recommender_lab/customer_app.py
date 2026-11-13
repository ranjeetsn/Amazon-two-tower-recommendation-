import os

import streamlit as st

from recommender_lab.llm_judge import judge_products
from recommender_lab.recommenders import RecommenderLab, load_products
from recommender_lab.storage import (
    init_db,
    mark_search_failed,
    record_feedback,
    record_query,
    record_recommendations,
)


st.set_page_config(page_title="Recommender Customer Simulator", layout="wide")
init_db()


@st.cache_resource
def get_lab():
    products_path = os.getenv("ESCI_PRODUCTS_PATH", "").strip()
    if not products_path:
        raise RuntimeError("Set ESCI_PRODUCTS_PATH before starting the app.")
    products = load_products(products_path)
    return RecommenderLab(products)


st.title("Product search: model comparison")
st.caption(
    "Local experiment. One query is saved to SQLite first, then scored by every "
    "available approach. The synthetic judge is an LLM relevance simulator, "
    "not a real customer."
)

if st.sidebar.button("Reload model artifacts"):
    get_lab.clear()
    st.rerun()

try:
    lab = get_lab()
except Exception as exc:
    st.error(str(exc))
    st.stop()

query = st.text_input("What product are you looking for?", key="query_input")
top_k = st.slider("Products per approach", min_value=3, max_value=20, value=10)

if st.button("Search all approaches", type="primary", disabled=not query.strip()):
    query_text = query.strip()
    st.session_state.pop("current_run", None)
    st.session_state.pop("current_ratings", None)
    request_id = None

    try:
        # Persist the input event before any model performs inference.
        request_id = record_query(query_text)
        results_by_model = lab.recommend_all(query_text, k=top_k)
        record_recommendations(
            request_id,
            results_by_model,
            model_version=lab.model_version,
        )
        st.session_state["current_run"] = {
            "request_id": request_id,
            "query": query_text,
            "results": results_by_model,
            "version": lab.model_version,
        }
    except Exception as exc:
        if request_id:
            mark_search_failed(request_id, exc)
            st.error(
                f"Query was saved as {request_id}, but recommendation generation "
                f"failed: {exc}"
            )
        else:
            st.error(f"Could not save the query: {exc}")

current = st.session_state.get("current_run")
if current:
    st.info(
        f"Saved query {current['request_id']} and recommendations using artifact "
        f"version {current['version']}."
    )

    methods = list(current["results"])
    tabs = st.tabs(methods)
    for tab, method in zip(tabs, methods):
        with tab:
            rows = current["results"][method]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.write("No candidates were returned.")

    if st.button("Simulate feedback with LLM", type="secondary"):
        unique_products = {}
        for rows in current["results"].values():
            for item in rows:
                unique_products.setdefault(
                    str(item["product_id"]),
                    {
                        "product_id": str(item["product_id"]),
                        "product_title": item["product_title"],
                    },
                )
        try:
            ratings = judge_products(
                current["query"],
                list(unique_products.values()),
            )
            record_feedback(
                current["request_id"],
                ratings,
                source="llm_simulated",
            )
            st.session_state["current_ratings"] = ratings
            st.success(
                f"Saved {len(ratings)} synthetic query-product labels to SQLite."
            )
        except Exception as exc:
            st.error(f"Synthetic feedback was not saved: {exc}")

    ratings = st.session_state.get("current_ratings")
    if ratings:
        st.subheader("Synthetic relevance labels")
        st.caption("E = exact · S = substitute · C = complement · I = irrelevant")
        st.dataframe(ratings, use_container_width=True, hide_index=True)
