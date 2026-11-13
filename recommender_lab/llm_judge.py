import json
import os

import requests


LABEL_GUIDE = """
Choose exactly one label for each product:
E = exact: directly satisfies the query.
S = substitute: a plausible alternative that satisfies the same need.
C = complement: related accessory/add-on, but does not itself satisfy the query.
I = irrelevant: unrelated or unsuitable.

Judge query-product relevance from the query and product title only. Do not infer personal taste.
Return a JSON object with a ratings array. Each entry must contain product_id, label, and a short rationale.
"""


def _parse_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def judge_products(query, products):
    endpoint = os.getenv("LLM_CHAT_COMPLETIONS_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not endpoint or not api_key or not model:
        raise RuntimeError(
            "Set LLM_CHAT_COMPLETIONS_URL, LLM_API_KEY, and LLM_MODEL to enable synthetic feedback."
        )

    # Judge each product once even when several recommenders returned it.
    unique = {}
    for item in products:
        product_id = str(item["product_id"])
        unique.setdefault(
            product_id,
            {
                "product_id": product_id,
                "product_title": str(item.get("product_title", ""))[:500],
            },
        )

    payload = {
        "query": str(query)[:1000],
        "products": list(unique.values())[:100],
    }

    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": LABEL_GUIDE},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        },
        timeout=90,
    )
    response.raise_for_status()

    response_text = response.json()["choices"][0]["message"]["content"]
    parsed = _parse_json(response_text)
    by_id = {
        str(item["product_id"]): item
        for item in parsed.get("ratings", [])
        if str(item.get("label", "")).upper() in {"E", "S", "C", "I"}
    }

    ratings = []
    for product_id, item in unique.items():
        rating = by_id.get(product_id)
        if rating is None:
            continue
        ratings.append(
            {
                **item,
                "label": str(rating["label"]).upper(),
                "rationale": str(rating.get("rationale", ""))[:500],
            }
        )

    return ratings
