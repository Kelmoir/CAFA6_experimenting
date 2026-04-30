

import os
import requests
import zipfile
from pathlib import Path
import torch
from Bio import SeqIO
import pandas as pd
import obonet
from tqdm.auto import tqdm as progressbar
import numpy as np
import datasets
from typing import List
import random

def get_data_from_github():
    """
    Ready the data from the github.
    """
    github_url = "https://github.com/Kelmoir/CAFA6_experimenting/raw/main/cafa-6-protein-function-prediction.zip"
    data_path =Path("data/")
    data_path.mkdir(exist_ok=True, parents=True)

    if (data_path/"Train").is_dir():
        print("Data already there, skipping download")
    else:
        print("Downlaoding data")
        with open(data_path/"cafa6.zip", "wb") as f:
            request = requests.get(github_url)
            print("Downloading data...")
            f.write(request.content)

        # unzip everything
        with zipfile.ZipFile(data_path/"cafa6.zip", "r") as zip_ref:
            print("unzipping files..")
            zip_ref.extractall(data_path)



def generate_dataset_subset(input_data: pd.DataFrame, label: str, entry_ids: list, prot_sequences: List[str], max_length=1024)-> datasets.Dataset:
  """
  Generates a dataset for a specific subset of EntryIDs to prevent leakage.
  """
  # Filter dataframe to only include specific proteins
  subset_df = input_data[input_data['EntryID'].isin(entry_ids)].reset_index(drop=True)

  label_idx = subset_df.columns.get_loc(label)+1
  label_list = []
  sequences = []
  features = []

  # Group by EntryID to aggregate aspects/terms
  grouped = subset_df.groupby('EntryID')

  for entry_id, group in progressbar(grouped, desc=f"Processing {label}"):
    if entry_id in prot_sequences and len(prot_sequences[entry_id]) <= max_length:
      unique_labels = " ".join(group[label].unique())
      features.append(entry_id)
      sequences.append(prot_sequences[entry_id])
      label_list.append(unique_labels)

  dataset_dict = {"protein": features, "labels": label_list, "sequence": sequences}
  return datasets.Dataset.from_dict(dataset_dict)

def generate_dataset_from_source(data_path: Path):
    # first: read in the data and translate it to internal representations for the model to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prot_sequences ={}
    unique_prot = []
    for sequence in SeqIO.parse(data_path/"Train/train_sequences.fasta", "fasta"):
        unique_prot.append(sequence.id.split("|")[1])
        prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X") # Reading the relevant data, and pre-formatting it for ProtBERT
    train_df = pd.read_csv(data_path/"Train/train_terms.tsv", delimiter="\t")
    obonet_graph = obonet.read_obo(data_path/"Train/go-basic.obo")
    unique_go_terms = np.unique(train_df["term"])
    unique_aspects = np.unique(train_df["aspect"])
    # generate id2label and label2id datasets for the model to use (now for aspects)
    id2label = {id: label for id, label in enumerate(unique_aspects)}
    label2id = {label: id for id, label in enumerate(unique_aspects)}

    # 1. Get unique protein IDs
    unique_protein_ids = list(train_df['EntryID'].unique())
    random.seed(42)
    random.shuffle(unique_protein_ids)

    # 2. Split IDs manually (80/20 split)
    test_size = int(len(unique_protein_ids) * 0.2)
    train_val_ids = unique_protein_ids[:-test_size]
    test_ids = unique_protein_ids[-test_size:]

    # 3. Split Train/Val IDs (70/30 of the 80%)
    val_size = int(len(train_val_ids) * 0.3)
    train_ids = train_val_ids[:-val_size]
    val_ids = train_val_ids[-val_size:]

    print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 4. Generate disjoint datasets
    processed_dataset = datasets.DatasetDict({
        "train": generate_dataset_subset(train_df, "aspect", train_ids),
        "validation": generate_dataset_subset(train_df, "aspect", val_ids),
        "test": generate_dataset_subset(train_df, "aspect", test_ids)
    })

    # 5. Apply transforms
    processed_dataset["train"] = processed_dataset["train"].with_transform(transform=preprocess_batch_partial)
    processed_dataset["validation"] = processed_dataset["validation"].with_transform(transform=preprocess_batch_partial)
    processed_dataset["test"] = processed_dataset["test"].with_transform(transform=preprocess_batch_partial)

    print("\nDisjoint datasets created successfully.")

