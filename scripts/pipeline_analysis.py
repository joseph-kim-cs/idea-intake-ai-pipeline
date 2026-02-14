import pandas as pd
from ibm_watsonx_ai.foundation_models import ModelInference
import os
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
PROJECT_ID = os.getenv('PROJECT_ID')
MODEL_ID = os.getenv('MODEL_ID')
URL = os.getenv('URL')

PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

model = ModelInference(
    model_id=MODEL_ID,
    params=PARAMS,
    credentials={
        "url": URL,
        "apikey": API_KEY
    },
    project_id=PROJECT_ID,
)

# sample data frame, we will make it modular in the future
df = pd.read_csv('data/cleaned_ideas.csv')

DESC_COL = "Idea Description"
CSV_PATH = 'data/cleaned_ideas.csv'
PROMPT_PATH = 'prompts/idea_summary_prompt.txt'

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cluster_ideas(df: pd.DataFrame, similarity_threshold: float = 0.78) -> pd.DataFrame:
    texts = df[DESC_COL].fillna("").astype(str).str.strip().tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=2500,
    )
    X = vectorizer.fit_transform(texts)

    sim = cosine_similarity(X)
    dist = 1.0 - sim

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
    )
    labels = clusterer.fit_predict(dist)
    df = df.copy()
    df["ClusterId"] = labels
    return df


def representative_idea(df: pd.DataFrame, sim_matrix: np.ndarray, idxs: list[int]) -> str:
    """Pick the idea with highest average similarity within the cluster."""
    if len(idxs) == 1:
        return df.loc[idxs[0], DESC_COL]
    sub = sim_matrix[np.ix_(idxs, idxs)]
    avg = sub.mean(axis=1)
    rep_idx = idxs[int(np.argmax(avg))]
    return df.loc[rep_idx, DESC_COL]


def build_input_object(df: pd.DataFrame, similarity_threshold: float = 0.78) -> dict:
    # Basic filtering
    df = df.copy()
    df[DESC_COL] = df[DESC_COL].fillna("").astype(str).str.strip()
    df = df[df[DESC_COL].str.len() >= 15]
    '''
    if "Is_Test_Or_Junk" in df.columns:
        df = df[~df["Is_Test_Or_Junk"]]
    '''
    df = df.reset_index(drop=True)

    # Vectorize once so we can compute representative ideas
    texts = df[DESC_COL].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=2500)
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)

    # Cluster
    dist = 1.0 - sim
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
    )
    labels = clusterer.fit_predict(dist)
    df["ClusterId"] = labels

    # Dataset summary
    top_domains = []
    if "Software Portfolio Domain" in df.columns:
        vc = df["Software Portfolio Domain"].dropna().astype(str).value_counts().head(5)
        top_domains = [{"domain": k, "count": int(v)} for k, v in vc.items()]

    top_products = []
    if "Product" in df.columns:
        vc = df["Product"].dropna().astype(str).value_counts().head(5)
        top_products = [{"product": k, "count": int(v)} for k, v in vc.items()]

    # Initiative candidates
    candidates = []
    for cid, grp in df.groupby("ClusterId"):
        idxs = grp.index.to_list()
        rep = representative_idea(df, sim, idxs)

        ex_ideas = grp[DESC_COL].head(8).tolist()

        cluster_domains = []
        if "Software Portfolio Domain" in grp.columns:
            cluster_domains = grp["Software Portfolio Domain"].dropna().astype(str).value_counts().head(3).index.tolist()

        cluster_products = []
        if "Product" in grp.columns:
            cluster_products = grp["Product"].dropna().astype(str).value_counts().head(3).index.tolist()

        candidates.append({
            "cluster_id": int(cid),
            "popularity_count": int(len(grp)),
            "representative_idea": rep,
            "example_ideas": ex_ideas,
            "top_domains": cluster_domains,
            "top_products": cluster_products
        })

    # Sort by popularity so the model sees “big themes” first
    candidates.sort(key=lambda x: x["popularity_count"], reverse=True)

    return {
        "dataset_summary": {
            "total_rows": int(len(df)),
            "total_clusters": int(df["ClusterId"].nunique()),
            "top_domains": top_domains,
            "top_products": top_products
        },
        "initiative_candidates": candidates
    }


def make_final_prompt(prompt_template: str, input_object: dict) -> str:
    input_json = json.dumps(input_object, ensure_ascii=False)
    return prompt_template.replace("{input_object}", input_json)



df = pd.read_csv(CSV_PATH)

input_object = build_input_object(df, similarity_threshold=0.78)

template = load_prompt_template(PROMPT_PATH)
final_prompt = make_final_prompt(template, input_object)

response = model.generate(final_prompt)
raw_text = response["results"][0]["generated_text"]
print(raw_text)