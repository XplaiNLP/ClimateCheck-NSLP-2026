from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os





cards_dataset = load_dataset('alextsiak/augmented_cards', split='train')
sm_dataset = load_dataset('Exorde/exorde-social-media-one-month-2024', split='train', streaming=True)

relevant_themes = ['Environment', 'Health', 'Politics', 'Science']

relevant_keywords = ['climate']

sm_dataset_en_gen = (row for row in sm_dataset if row["language"] == "en" and row["primary_theme"] in relevant_themes 
        and any(kw in row["english_keywords"].lower()
        for kw in relevant_keywords)
        )

#prepare embedding model

emb_model = SentenceTransformer('intfloat/e5-large-v2')

#prepare claims from CARDS
seed_texts = [claim['text'] 
              for claim in cards_dataset 
              if claim.get('acards_claim') 
              and claim['acards_claim'] != '0_0'
             ]

#narrative ids of CARDS

seed_ids = [claim["acards_claim"] 
            for claim in cards_dataset
            if claim.get('acards_claim') 
            and claim['acards_claim'] != '0_0'
           ]

# embed CARDS claims
seed_embeddings = emb_model.encode(seed_texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

#finding a threshold

max_posts = 1000
batch_size = 128
texts = []
rows = []

for row in sm_dataset_en_gen:
    texts.append(row["original_text"])
    rows.append(row)
    if len(texts) == max_posts:
        break


embeddings = emb_model.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

#compute similarities
similarities = embeddings @ seed_embeddings.T
max_sim = np.max(similarities, axis=1)

# embed exorde claims

batch_size = 128
texts_batch = []
rows_batch = []
similarity_threshold = 0.86

for row in sm_dataset_en_gen:
    texts_batch.append(row["original_text"])
    rows_batch.append(row)  # keep full row for later

    if len(texts_batch) == batch_size:
        embeddings = emb_model.encode(
            texts_batch,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        #cosine similarity
        similarities = embeddings @ seed_embeddings.T 
        #most similar CARDS claim
        nearest_seed_idx = np.argmax(similarities, axis=1)
        max_sim = np.max(similarities, axis=1)
        #narrative_id of nearest CARDS claim for sm_dataset row
        labeled_rows = []
        inspection_rows = []

        for row, seed_idx, sim in zip(rows_batch, nearest_seed_idx, max_sim):
            seed_label = seed_ids[seed_idx]
            seed_text = seed_texts[seed_idx]

            #inspection data (ALWAYS saved)
            inspection_rows.append({
                "sm_text": row["original_text"],
                "similarity": float(sim),
                "matched_seed_text": seed_text,
                "matched_seed_label": seed_label,
            })

            #weak labels (ONLY if above threshold)
            if sim >= similarity_threshold:
                labeled_row = dict(row)
                labeled_row["narrative_id"] = seed_label
                labeled_row["similarity"] = float(sim)
                labeled_rows.append(labeled_row)

        # save weak labels
        if labeled_rows:
            pd.DataFrame(labeled_rows).to_csv(
                "weak_labels.csv",
                mode="a",
                index=False,
                header=not os.path.exists("weak_labels.csv"),
            )

        # save similarity inspection data
        if inspection_rows:
            pd.DataFrame(inspection_rows).to_csv(
                "similarity_inspection.csv",
                mode="a",
                index=False,
                header=not os.path.exists("similarity_inspection.csv"),
            )

        texts_batch = []
        rows_batch = []

