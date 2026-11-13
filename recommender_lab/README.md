# Local ESCI recommender lab

This experiment adds lexical, supervised-ranking, and two-tower approaches beside the existing demo without changing its FastAPI service.

## Local architecture

- Customer Streamlit app writes each query to SQLite before calling any ranker, then stores each model's displayed products and marks the search complete. A failed ranking attempt remains logged with its error.
- An optional LLM judge assigns ESCI-style labels to the unique products shown for that query. These are synthetic relevance labels, not real customer outcomes.
- Model training runs only when you execute the training notebook. It reads the Kaggle training data plus saved feedback from SQLite. After 1,000 new unique labeled query-product pairs, the dashboard marks retraining as ready.
- Dashboard Streamlit app reads the same SQLite database and reports recommendation and feedback results.
- Kafka is intentionally not required for this single-machine simulation. SQLite is the durable interaction log; the notebook reads it when you choose to retrain. Kafka may be useful later for multiple independent consumers, replay, or distributed deployment.

## Data

Set paths to the Kaggle ESCI files. Expected columns include:

- Examples: \`query_id\`, \`query\`, \`product_id\`, \`esci_label\` (E/S/C/I)
- Products: \`product_id\`, \`product_title\`; optional \`product_brand\`

The loader accepts CSV and Parquet. Keep the provided official train/test split intact; the training script trains on rows with split=train when that column exists.

## Setup (Python 3.10)

The existing repository pins TensorFlow 2.15, so use the repository's Python 3.10 environment.

\`\`\`powershell
python -m pip install -r recommender_lab/requirements.txt
$env:ESCI_EXAMPLES_PATH = "C:\data\shopping_queries_dataset_examples.parquet"
$env:ESCI_PRODUCTS_PATH = "C:\data\shopping_queries_dataset_products.parquet"
$env:RECOMMENDER_DB_PATH = "recommender_lab\data\interactions.sqlite3"
$env:RECOMMENDER_ARTIFACT_DIR = "recommender_lab\artifacts"
\`\`\`

For LLM-generated simulation labels, configure an OpenAI-compatible chat-completions endpoint:

\`\`\`powershell
$env:LLM_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
$env:LLM_API_KEY = "<your key>"
$env:LLM_MODEL = "<your model>"
\`\`\`

The API key is read from the environment and is never written to SQLite.

## Run

Open recommender_lab/training_workbench.ipynb in VS Code or Jupyter with the project's Python 3.10 environment. Update the dataset paths in its first code cell and run the notebook. It trains the initial models when no active model exists. Later, it trains again when SQLite has at least 1,000 new unique query-product labels. Set FORCE_TRAIN = True in the notebook to intentionally override that threshold.

Start the two Streamlit apps in separate terminals:

- streamlit run recommender_lab/customer_app.py
- streamlit run recommender_lab/dashboard_app.py

The customer app shows results from token overlap, fuzzy matching, character TF-IDF, word TF-IDF, BM25, XGBoost, and the two-tower model when their artifacts are available. Use its synthetic-feedback button to have the configured LLM label displayed products.

The notebook snapshots the feedback count at training start and records it after a successful run. Feedback arriving while training is in progress remains pending for the next notebook run.

## Label and evaluation notes

- E = exact match, S = substitute, C = complement, I = irrelevant.
- The LLM simulation evaluates product-to-query relevance; it cannot represent personal taste or actual purchase satisfaction.
- XGBoost uses graded relevance E=3, S=2, C=1, I=0. The two-tower starter model uses E as positive and I as negative, leaving S/C out rather than treating them as definite negatives.
- Use actual user impressions and outcomes for production retraining. An unseen product is not a negative example.
