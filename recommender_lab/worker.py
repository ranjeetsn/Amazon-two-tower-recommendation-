import os
import time
import traceback

from recommender_lab.storage import (
    feedback_count,
    init_db,
    last_trained_feedback_count,
    load_training_feedback,
    mark_training_finished,
    mark_training_started,
)
from recommender_lab.train import train_from_files


def run_worker():
    init_db()
    threshold = int(os.getenv("RETRAIN_THRESHOLD", "1000"))
    poll_seconds = int(os.getenv("RETRAIN_POLL_SECONDS", "20"))
    examples_path = os.getenv("ESCI_EXAMPLES_PATH", "").strip()
    products_path = os.getenv("ESCI_PRODUCTS_PATH", "").strip()
    artifact_root = os.getenv(
        "RECOMMENDER_ARTIFACT_DIR", "recommender_lab/artifacts"
    )

    if not examples_path or not products_path:
        raise RuntimeError("Set ESCI_EXAMPLES_PATH and ESCI_PRODUCTS_PATH first.")

    print(
        f"Retraining worker is watching SQLite for {threshold} new labeled pairs."
    )
    while True:
        count_at_start = feedback_count()
        pending = count_at_start - last_trained_feedback_count()

        if pending >= threshold:
            run_id = mark_training_started(pending)
            try:
                feedback = load_training_feedback()
                version = train_from_files(
                    examples_path,
                    products_path,
                    feedback_df=feedback,
                    artifact_root=artifact_root,
                )
                mark_training_finished(
                    run_id,
                    "succeeded",
                    model_version=version,
                    message=f"Trained with {len(feedback)} unique feedback pairs.",
                    trained_through_feedback_count=count_at_start,
                )
                print(f"Published model version {version}.")
            except Exception as exc:
                detail = f"{exc}\n{traceback.format_exc()}"
                mark_training_finished(
                    run_id,
                    "failed",
                    message=detail,
                )
                print(f"Training failed: {exc}")
                time.sleep(max(poll_seconds, 300))
                continue

        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_worker()
