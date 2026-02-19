import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def split_labels(example, label_col):
    example[label_col] = example[label_col].split(";")
    return example


def main():
    dataset_cc = load_dataset("rabuahmad/climatecheck", split="train")
    dataset_ca = load_dataset("alextsiak/augmented_cards", split="train")

    df_cc = dataset_cc.to_pandas()
    df_cc_unique = df_cc.drop_duplicates(subset="claim_id", keep="first")
    dataset_cc = Dataset.from_pandas(df_cc_unique)

    dataset_cc = dataset_cc.map(lambda x: split_labels(x, "narrative"))
    dataset_ca = dataset_ca.map(lambda x: split_labels(x, "claim"))

    #Identify rare classes in CC26
    all_labels_cc = [lbl for row in dataset_cc["narrative"] for lbl in row]
    label_counts_cc = Counter(all_labels_cc)

    min_count = 100 #can be changed
    maj_class = "0_0"

    rare_classes = [
        label
        for label, count in label_counts_cc.items()
        if count < min_count and label != maj_class
    ]

    print("Rare classes:", rare_classes)

    #check which rare classes appear in CARDS
    cards_df = dataset_ca.to_pandas()
    all_labels_cards = [lbl for row in cards_df["claim"] for lbl in row]
    label_counts_cards = Counter(all_labels_cards)

    augmentable_classes = []
    non_augmentable_classes = []

    for label in rare_classes:
        if label in label_counts_cards:
            augmentable_classes.append(label)
        else:
            non_augmentable_classes.append(label)

    print("Augmentable classes:", augmentable_classes)
    print("Non-augmentable classes:", non_augmentable_classes)


    dataset_ca = dataset_ca.rename_column("claim", "narrative")
    dataset_ca = dataset_ca.rename_column("text", "claim")

    #Sample augmentation examples
    augmented_examples = []

    for label in augmentable_classes:
        candidates = [ex for ex in dataset_ca if label in ex["narrative"]]
        num_to_add = min(300, len(candidates))

        sampled = random.sample(candidates, num_to_add)
        augmented_examples.extend(sampled)

    #Concatenate and shuffle datasets
    extra_dataset_aug = Dataset.from_list(augmented_examples)
    dataset_cc_augmented = concatenate_datasets(
        [dataset_cc, extra_dataset_aug]
    )

    print("Original dataset size:", len(dataset_cc))
    print("Augmented dataset size:", len(dataset_cc_augmented))

    dataset_aug = dataset_cc_augmented.shuffle(seed=42)

    print("Columns:", dataset_aug.column_names)
    print("Sample examples:")
    for i in range(5):
        print(dataset_aug[i])

    print("Total examples:", len(dataset_aug))

    #Label distribution after augmentation
    all_labels_aug = [
        lbl for labels in dataset_aug["narrative"] for lbl in labels
    ]
    label_counts_aug = Counter(all_labels_aug)

    print("\nLabel distribution:")
    for label, count in label_counts_aug.most_common():
        print(f"{label}: {count}")

    if "__index_level_0__" in dataset_aug.column_names:
        dataset_aug = dataset_aug.remove_columns("__index_level_0__")


if __name__ == "__main__":
    main()
