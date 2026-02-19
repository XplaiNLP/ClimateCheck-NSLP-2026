from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

dataset_ex = load_dataset('alextsiak/exorde_CARDS_labels', split='train')

#run model on exorde dataset to confirm weak labels

pipe = pipeline("text-classification", model='crarojasca/BinaryAugmentedCARDS', device=0) # device=0 uses GPU
texts = list(dataset_ex['original_text'])
results = pipe(texts, batch_size=16)


#extract labels and clean them
numeric_labels = [res['label'].split('_')[-1] if 'LABEL_' in res['label'] else res['label'] for res in results]
confidences = [round(res['score'], 4) for res in results]

df = pd.DataFrame({
    'claim': texts,
    'prediction': numeric_labels,
    'narrative_id': dataset_ex['narrative_id'] # Keeping your original ID column
})

disinfo_df = df[df['prediction'] == 'Contrarian'].copy()
disinfo_df = disinfo_df.drop(columns=['prediction'])


disinfo_df.to_csv('exorde_disinfo_claims.csv', index=False)
